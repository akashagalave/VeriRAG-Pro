"""
evaluate.py
-----------
DeepEval evaluation pipeline with regression detection.

Two modes:
  1. Full evaluation (default):
     - Loads or generates golden Q&A pairs from the PDF
     - Runs all test cases through the RAG pipeline
     - Saves scores to eval_results.json
     - Compares against baseline (eval_baseline.json)
     - Exits with code 1 if any metric drops below threshold
     - Updates baseline on success

  2. CI/CD gate (--ci flag):
     - Same as above but writes a GitHub Actions summary
     - Designed to run as a blocking job before EKS deployment

Usage:
  python evaluate.py              # full evaluation, update baseline if pass
  python evaluate.py --ci         # CI mode — exit 1 on regression
  python evaluate.py --generate   # regenerate goldens.json from PDF

Exit codes:
  0 — all metrics at or above threshold (safe to deploy)
  1 — regression detected or evaluation error (block deployment)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

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
from backend.vector_store import add_paper

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

PDF_PATH             = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE         = Path("goldens.json")
RESULTS_FILE         = Path("eval_results.json")
BASELINE_FILE        = Path("eval_baseline.json")
SUMMARY_FILE         = Path("eval_summary.md")   # written in CI mode

MAX_CONTEXTS         = 5
GOLDENS_PER_CONTEXT  = 2
METRIC_THRESHOLD     = 0.7
JUDGE_MODEL          = "gpt-4o-mini"

# Metrics where a drop of more than this delta blocks deployment
REGRESSION_TOLERANCE = 0.05   # 5 percentage points

# Metrics that must NEVER drop below threshold (hard gates)
HARD_GATE_METRICS = {"Faithfulness", "Answer Relevancy"}


# ── Golden generation ─────────────────────────────────────────────────────────

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


# ── RAG pipeline runner ───────────────────────────────────────────────────────

def run_rag_query(graph, query: str, session_id: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": session_id}}
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "query": query,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [
        doc.page_content for doc in (final_state.get("retrieved_docs") or [])
    ]
    return answer, retrieval_context


# ── Score extraction ──────────────────────────────────────────────────────────

def extract_scores(results) -> dict[str, float]:
    """
    Aggregate metric scores across all test cases.
    Returns {metric_name: average_score}.
    """
    score_buckets: dict[str, list[float]] = {}
    for test_result in results.test_results:
        for m in test_result.metrics_data:
            score_buckets.setdefault(m.name, []).append(m.score or 0.0)
    return {name: sum(scores) / len(scores) for name, scores in score_buckets.items()}


# ── Regression detection ──────────────────────────────────────────────────────

def detect_regression(
    current_scores: dict[str, float],
    baseline_scores: dict[str, float],
) -> list[str]:
    """
    Returns a list of regression messages (empty list = no regression).

    Regression rules:
      1. Any HARD_GATE metric below METRIC_THRESHOLD → regression.
      2. Any metric drops more than REGRESSION_TOLERANCE vs baseline → regression.
    """
    failures: list[str] = []

    for metric, score in current_scores.items():
        # Hard gate: must stay above threshold
        if metric in HARD_GATE_METRICS and score < METRIC_THRESHOLD:
            failures.append(
                f"HARD GATE FAIL — {metric}: {score:.3f} < threshold {METRIC_THRESHOLD}"
            )
            continue

        # Regression vs baseline
        baseline = baseline_scores.get(metric)
        if baseline is not None:
            drop = baseline - score
            if drop > REGRESSION_TOLERANCE:
                failures.append(
                    f"REGRESSION — {metric}: {score:.3f} (baseline={baseline:.3f}, "
                    f"drop={drop:.3f} > tolerance={REGRESSION_TOLERANCE})"
                )

    return failures


# ── Markdown summary (for GitHub Actions) ────────────────────────────────────

def write_summary(
    current_scores: dict[str, float],
    baseline_scores: dict[str, float],
    failures: list[str],
    ci_mode: bool,
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

    if failures:
        lines.append("\n## ❌ Regressions Detected\n")
        for f in failures:
            lines.append(f"- {f}")
        lines.append(
            f"\n**Deployment BLOCKED** — fix regressions before merging."
        )
    else:
        lines.append("\n## ✅ All Checks Passed\n")
        lines.append("Safe to deploy.")

    summary_text = "\n".join(lines)
    SUMMARY_FILE.write_text(summary_text, encoding="utf-8")
    print(summary_text)

    # Write to GitHub Actions step summary if available
    if ci_mode:
        gh_summary = os.getenv("GITHUB_STEP_SUMMARY")
        if gh_summary:
            with open(gh_summary, "a", encoding="utf-8") as f:
                f.write(summary_text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="VeriRAG DeepEval regression pipeline")
    parser.add_argument("--ci",       action="store_true", help="CI/CD mode — exit 1 on regression")
    parser.add_argument("--generate", action="store_true", help="Regenerate golden pairs from PDF")
    args = parser.parse_args()

    # Load or generate goldens
    if args.generate or not GOLDENS_FILE.exists():
        pairs = generate_goldens()
    else:
        pairs = load_goldens()

    logger.info("Running evaluation on %d golden pairs…", len(pairs))

    # Load baseline scores
    baseline_scores: dict[str, float] = {}
    if BASELINE_FILE.exists():
        try:
            baseline_scores = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
            logger.info("Loaded baseline scores from %s", BASELINE_FILE)
        except Exception as exc:
            logger.warning("Could not load baseline: %s", exc)

    # Build graph and load docs into a SINGLE shared eval session
    docs = load_document(PDF_PATH)
    graph = build_graph(db_path="eval_checkpoints.db")
    eval_session_id = f"eval_{uuid4().hex}"
    add_paper(docs, eval_session_id)
    logger.info("Evaluation session: %s (%d chunks)", eval_session_id, len(docs))

    # Build metrics
    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD,  model=JUDGE_MODEL),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD,     model=JUDGE_MODEL),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD,  model=JUDGE_MODEL),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD,      model=JUDGE_MODEL),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD,         model=JUDGE_MODEL),
    ]

    # Build test cases
    test_cases: list[LLMTestCase] = []
    for pair in pairs:
        query = pair["input"] + " as per the report in knowledge base"
        answer, retrieval_context = run_rag_query(graph, query, eval_session_id)
        test_cases.append(LLMTestCase(
            input=pair["input"],
            actual_output=answer,
            expected_output=pair["expected_output"],
            retrieval_context=retrieval_context,
        ))

    # Run evaluation
    results = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(max_concurrent=3, throttle_value=5),
    )

    # Save full results
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

    # Extract aggregate scores and detect regressions
    current_scores = extract_scores(results)
    failures = detect_regression(current_scores, baseline_scores)

    # Write markdown summary
    write_summary(current_scores, baseline_scores, failures, ci_mode=args.ci)

    if failures:
        logger.error("EVALUATION GATE FAILED — %d regression(s) detected:", len(failures))
        for f in failures:
            logger.error("  %s", f)
        return 1

    # All passed — update baseline
    BASELINE_FILE.write_text(
        json.dumps(current_scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Baseline updated → %s", BASELINE_FILE)
    logger.info("✅ Evaluation passed. Safe to deploy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
