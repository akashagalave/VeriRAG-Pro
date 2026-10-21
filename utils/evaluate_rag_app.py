"""Evaluate saved VeriRAG-Pro Attention-paper answers with DeepEval 4.x.

This script never calls LangGraph or ingests the PDF again.
Run: python evaluate_rag_app.py --limit 1
     python evaluate_rag_app.py
"""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, SingleTurnParams
from config.parameter_config import params_config


ROOT_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "evaluation"
    / "eval_dataset"
    / f"{params_config.evaluation_dataset.evaluation_dataset_filename}.json"
)

REPORT_DIR = (
    ROOT_DIR / params_config.evaluation.display_config.results_dir
)

JUDGE_MODEL = params_config.evaluation.judge_llm

MAX_CONCURRENT = (
    params_config.evaluation.async_config.max_concurrent
)

THROTTLE_SECONDS = (
    params_config.evaluation.async_config.throttle_value
)

# Diagnostic threshold only, not an automatic regression gate
THRESHOLD = 0.7


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("attention-eval")
load_dotenv(ROOT_DIR / ".env")


def load_cases(path: Path, limit: int = 0) -> list[LLMTestCase]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation dataset must be a nonempty JSON array")
    if limit:
        rows = rows[:limit]

    cases: list[LLMTestCase] = []
    for index, row in enumerate(rows, 1):
        required = ("input", "actual_output", "expected_output", "retrieval_context")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Case {index}: missing {missing}")
        context = row["retrieval_context"]
        if not isinstance(context, list) or not context or not all(
            isinstance(chunk, str) and chunk.strip() for chunk in context
        ):
            raise ValueError(f"Case {index}: retrieval_context must contain text chunks")
        if any(not isinstance(row[key], str) or not row[key].strip()
               for key in ("input", "actual_output", "expected_output")):
            raise ValueError(f"Case {index}: input/actual/expected must be nonempty strings")
        if row.get("route") not in (None, "retrieve"):
            raise ValueError(f"Case {index}: expected retrieval route, got {row['route']!r}")
        cases.append(LLMTestCase(
            input=row["input"],
            actual_output=row["actual_output"],
            expected_output=row["expected_output"],
            retrieval_context=context,
        ))
    return cases


def build_metrics():
    common = {"model": JUDGE_MODEL, "threshold": THRESHOLD}
    correctness = GEval(
        name="Answer Correctness",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        criteria=("Compare the actual answer with the reference answer for factual accuracy. "
                  "Penalize contradictions and missing essential facts; do not penalize harmless paraphrases."),
        rubric=[
            Rubric(score_range=(0, 3), expected_outcome="Incorrect or seriously misleading"),
            Rubric(score_range=(4, 6), expected_outcome="Partially correct with important omissions"),
            Rubric(score_range=(7, 9), expected_outcome="Mostly correct, minor omissions"),
            Rubric(score_range=(10, 10), expected_outcome="Accurate and complete"),
        ],
        **common,
    )
    explanation = GEval(
        name="Simple Explanation",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Check if the response directly explains the question in a clear, logical order.",
            "Prefer understandable explanations, while allowing necessary technical terminology.",
            "Penalize unexplained jargon and confusing or needlessly repetitive wording.",
        ],
        rubric=[
            Rubric(score_range=(0, 3), expected_outcome="Confusing and hard to follow"),
            Rubric(score_range=(4, 7), expected_outcome="Understandable but needs improvement"),
            Rubric(score_range=(8, 10), expected_outcome="Clear, accessible, technically appropriate"),
        ],
        **common,
    )
    return [
        ContextualPrecisionMetric(**common),
        ContextualRecallMetric(**common),
        ContextualRelevancyMetric(**common),
        AnswerRelevancyMetric(**common),
        FaithfulnessMetric(**common),
        correctness,
        explanation,
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved Attention-paper RAG answers")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate first N cases; 0 = all 10")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY not set in environment or .env")

    cases = load_cases(DATASET_PATH, args.limit)
    logger.info("Loaded %d saved evaluation cases (no RAG regeneration)", len(cases))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result = evaluate(
        test_cases=cases,
        metrics=build_metrics(),
        async_config=AsyncConfig(max_concurrent=MAX_CONCURRENT, throttle_value=THROTTLE_SECONDS),
        display_config=DisplayConfig(results_folder=str(REPORT_DIR), print_results=True),
    )

    records = []
    for case in result.test_results:
        records.append({
            "input": case.input,
            "actual_output": case.actual_output,
            "success": case.success,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "success": m.success,
                    "reason": m.reason,
                    "error": str(m.error) if getattr(m, "error", None) else None,
                }
                for m in (case.metrics_data or [])
            ],
        })
    filename = "attention_metrics_smoke.json" if args.limit else "attention_metrics.json"
    output_path = REPORT_DIR / filename
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved metric results: {output_path}")
    for record in records:
        print("PASS" if record["success"] else "REVIEW", "-", record["input"][:85])
    print("This is a local diagnostic evaluation, not a CI/CD regression gate.")


if __name__ == "__main__":
    main()
