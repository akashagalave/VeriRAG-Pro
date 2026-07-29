import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, get_collection_name, qdrant_client

load_dotenv()


PDF_PATH             = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE         = Path("goldens.json")
RESULTS_FILE         = Path("eval_results.json")
BASELINE_FILE        = Path("eval_baseline.json")
SUMMARY_FILE         = Path("eval_summary.md")
CHECKPOINT_DB        = Path("eval_checkpoints.db")

MAX_CONTEXTS         = 5
GOLDENS_PER_CONTEXT  = 2
METRIC_THRESHOLD     = 0.7
JUDGE_MODEL          = "gpt-4o-mini"

REGRESSION_TOLERANCE = 0.05

HARD_GATE_METRICS = {"Faithfulness", "Answer Relevancy"}

EVAL_SESSION_ID = "eval_fixed_session"

RETRIEVAL_BIAS_SUFFIX = " as per the report in knowledge base"


def generate_goldens() -> list[dict]:
    synthesizer = Synthesizer()
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[PDF_PATH],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=MAX_CONTEXTS,
        ),
    )
    pairs = [
        {"input": g.input, "expected_output": g.expected_output}
        for g in goldens
        if g.input and g.expected_output
    ]
    GOLDENS_FILE.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Generated %d golden pairs → %s", len(pairs), GOLDENS_FILE)
    return pairs


def load_goldens() -> list[dict]:
    return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))


def reset_eval_collection(session_id: str) -> None:
    collection_name = get_collection_name(session_id)
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
        logger.info("Deleted stale eval collection: %s", collection_name)


def cleanup_eval_collection(session_id: str) -> None:
    collection_name = get_collection_name(session_id)
    try:
        if qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name)
            logger.info("Cleaned up eval collection: %s", collection_name)
    except Exception as exc:
        logger.warning("Could not clean up eval collection %s: %s", collection_name, exc)


def run_rag_query(
    graph,
    query: str,
    session_id: str,
    thread_id: str,
) -> tuple[str, list[str], str]:
    config = {"configurable": {"thread_id": thread_id}}
    final_state = graph.invoke(
        {
            # Full initial state, matching what api.py sends — so the eval
            # exercises the same code path as production.
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "query": query,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [
        doc.page_content for doc in (final_state.get("retrieved_docs") or [])
    ]
    route = final_state.get("route") or "unknown"
    return answer, retrieval_context, route


def extract_scores(results) -> dict[str, float]:
    score_buckets: dict[str, list[float]] = {}
    for test_result in results.test_results:
        for m in test_result.metrics_data:
            score_buckets.setdefault(m.name, []).append(m.score or 0.0)
    return {name: sum(scores) / len(scores) for name, scores in score_buckets.items()}


def detect_regression(
    current_scores: dict[str, float],
    baseline_scores: dict[str, float],
) -> list[str]:
    failures: list[str] = []

    for metric, score in current_scores.items():
        if metric in HARD_GATE_METRICS and score < METRIC_THRESHOLD:
            failures.append(
                f"HARD GATE FAIL — {metric}: {score:.3f} < threshold {METRIC_THRESHOLD}"
            )
            continue

        baseline = baseline_scores.get(metric)
        if baseline is not None:
            drop = baseline - score
            if drop > REGRESSION_TOLERANCE:
                failures.append(
                    f"REGRESSION — {metric}: {score:.3f} (baseline={baseline:.3f}, "
                    f"drop={drop:.3f} > tolerance={REGRESSION_TOLERANCE})"
                )

    return failures


def write_summary(
    current_scores: dict[str, float],
    baseline_scores: dict[str, float],
    failures: list[str],
    ci_mode: bool,
    route_counts: dict[str, int] | None = None,
    empty_context_goldens: list[int] | None = None,
) -> None:
    lines = ["# VeriRAG Evaluation Report\n"]
    lines.append("| Metric | Score | Baseline | Status |")
    lines.append("|--------|-------|----------|--------|")

    for metric, score in sorted(current_scores.items()):
        baseline = baseline_scores.get(metric, "—")
        if isinstance(baseline, float):
            drop = baseline - score
            if drop > REGRESSION_TOLERANCE:
                status = "🔴 REGRESSION"
            elif score < METRIC_THRESHOLD:
                status = "🟡 BELOW THRESHOLD"
            else:
                status = "✅ PASS"
            baseline_str = f"{baseline:.3f}"
        else:
            status = "✅ PASS" if score >= METRIC_THRESHOLD else "🟡 BELOW THRESHOLD"
            baseline_str = "—"
        lines.append(f"| {metric} | {score:.3f} | {baseline_str} | {status} |")
    if route_counts:
        lines.append("\n## Route Distribution\n")
        lines.append("| Route | Goldens |")
        lines.append("|-------|---------|")
        for route, count in sorted(route_counts.items()):
            lines.append(f"| {route} | {count} |")
        non_retrieve = sum(v for k, v in route_counts.items() if k != "retrieve")
        if non_retrieve:
            lines.append(
                f"\n⚠️ {non_retrieve} golden(s) did not take the `retrieve` route. "
                "Context metrics score 0 for those cases."
            )

    if empty_context_goldens:
        lines.append(
            f"\n⚠️ **{len(empty_context_goldens)} golden(s) produced an empty "
            f"retrieval_context**: indices {empty_context_goldens}. "
            "All four context metrics score 0 for these."
        )

    if failures:
        lines.append("\n## ❌ Regressions Detected\n")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("\n**Deployment BLOCKED** — fix regressions before merging.")
    else:
        lines.append("\n## ✅ All Checks Passed\n")
        lines.append("Safe to deploy.")

    summary_text = "\n".join(lines)
    SUMMARY_FILE.write_text(summary_text, encoding="utf-8")
    print(summary_text)
    if ci_mode:
        gh_summary = os.getenv("GITHUB_STEP_SUMMARY")
        if gh_summary:
            with open(gh_summary, "a", encoding="utf-8") as f:
                f.write(summary_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="VeriRAG DeepEval regression pipeline")
    parser.add_argument("--ci",       action="store_true", help="CI/CD mode — exit 1 on regression")
    parser.add_argument("--generate", action="store_true", help="Regenerate golden pairs from PDF")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only run the first N goldens — for cheap smoke runs. 0 = all.",
    )
    args = parser.parse_args()

    if args.generate or not GOLDENS_FILE.exists():
        pairs = generate_goldens()
    else:
        pairs = load_goldens()

    if args.limit > 0:
        pairs = pairs[: args.limit]
        logger.warning(
            "--limit %d active: running a SUBSET. Do not write this to the baseline.",
            args.limit,
        )

    logger.info("Running evaluation on %d golden pairs…", len(pairs))

    baseline_scores: dict[str, float] = {}
    if BASELINE_FILE.exists():
        try:
            baseline_scores = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
            logger.info("Loaded baseline scores from %s", BASELINE_FILE)
        except Exception as exc:
            logger.warning("Could not load baseline: %s", exc)

    reset_eval_collection(EVAL_SESSION_ID)
    CHECKPOINT_DB.unlink(missing_ok=True)
    _eval_conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _eval_checkpointer = SqliteSaver(_eval_conn)
    graph = build_graph(checkpointer=_eval_checkpointer)

    try:
        docs = load_document(PDF_PATH)
        add_paper(docs, EVAL_SESSION_ID)
        logger.info("Evaluation session: %s (%d chunks)", EVAL_SESSION_ID, len(docs))

        metrics = [
            ContextualPrecisionMetric(threshold=METRIC_THRESHOLD,  model=JUDGE_MODEL),
            ContextualRecallMetric(threshold=METRIC_THRESHOLD,     model=JUDGE_MODEL),
            ContextualRelevancyMetric(threshold=METRIC_THRESHOLD,  model=JUDGE_MODEL),
            AnswerRelevancyMetric(threshold=METRIC_THRESHOLD,      model=JUDGE_MODEL),
            FaithfulnessMetric(threshold=METRIC_THRESHOLD,         model=JUDGE_MODEL),
        ]

        test_cases: list[LLMTestCase] = []
        route_counts: dict[str, int] = {}
        empty_context_goldens: list[int] = []

        for idx, pair in enumerate(pairs):
            asked_query = pair["input"] + RETRIEVAL_BIAS_SUFFIX

            # One isolated checkpoint thread per golden — the core fix.
            thread_id = f"{EVAL_SESSION_ID}__golden_{idx:03d}"

            answer, retrieval_context, route = run_rag_query(
                graph, asked_query, EVAL_SESSION_ID, thread_id
            )

            route_counts[route] = route_counts.get(route, 0) + 1

            if not retrieval_context:
                empty_context_goldens.append(idx)
                logger.warning(
                    "Golden %d produced EMPTY retrieval_context (route=%s) — "
                    "all four context metrics will score 0 for this case.",
                    idx, route,
                )

            logger.info(
                "Golden %d/%d done | route=%s | chunks=%d | thread=%s",
                idx + 1, len(pairs), route, len(retrieval_context), thread_id,
            )

            test_cases.append(LLMTestCase(
                input=asked_query,
                actual_output=answer,
                expected_output=pair["expected_output"],
                retrieval_context=retrieval_context,
            ))

        logger.info("Route distribution across %d goldens: %s", len(pairs), route_counts)
        if empty_context_goldens:
            logger.warning(
                "%d/%d goldens had empty retrieval_context: %s",
                len(empty_context_goldens), len(pairs), empty_context_goldens,
            )

        results = evaluate(
            test_cases,
            metrics,
            async_config=AsyncConfig(max_concurrent=3, throttle_value=5),
        )

        summary_rows = []
        for test_result in results.test_results:
            summary_rows.append({
                "input": test_result.input,
                "actual_output": test_result.actual_output,
                "success": test_result.success,
                "metrics": [
                    {
                        "name": m.name,
                        "score": m.score,
                        "passed": m.success,
                        "reason": m.reason,
                    }
                    for m in test_result.metrics_data
                ],
            })
        RESULTS_FILE.write_text(
            json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Full results saved → %s", RESULTS_FILE)

        current_scores = extract_scores(results)
        failures = detect_regression(current_scores, baseline_scores)

        write_summary(
            current_scores, baseline_scores, failures,
            ci_mode=args.ci,
            route_counts=route_counts,
            empty_context_goldens=empty_context_goldens,
        )

        if failures:
            logger.error("EVALUATION GATE FAILED — %d regression(s) detected:", len(failures))
            for f in failures:
                logger.error("  %s", f)
            return 1

        if args.limit > 0:
            logger.warning(
                "Skipping baseline write — this was a --limit %d subset run.", args.limit
            )
        else:
            BASELINE_FILE.write_text(
                json.dumps(current_scores, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("Baseline updated → %s", BASELINE_FILE)

        logger.info("✅ Evaluation passed. Safe to deploy.")
        return 0

    finally:
        cleanup_eval_collection(EVAL_SESSION_ID)
        try:
            _eval_conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
