import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDENS = ROOT / "data/evaluation/goldens/attention_goldens.json"
DEFAULT_BASELINE = ROOT / "reports/evaluation/attention_metrics_k3_baseline.json"
DEFAULT_CANDIDATE = ROOT / "reports/evaluation/attention_metrics.json"
DEFAULT_REPORT = ROOT / "reports/evaluation/regression_summary.md"
DEFAULT_TOLERANCE = 0.15  # maximum allowed drop in any metric score
EXPECTED_METRICS = {
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
    "answer_relevancy",
    "faithfulness",
    "answer_correctness_geval",
    "simple_explanation_geval",
}


def read_list(path: Path, label: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list) or not obj or not all(isinstance(x, dict) for x in obj):
        raise ValueError(f"{label} must be a nonempty JSON list of objects: {path}")
    return obj


def metric_key(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")



def validate_and_summarize(
    report: list[dict],
    golden_inputs: list[str],
    label: str,
) -> dict[str, float]:

    actual_inputs = [row.get("input") for row in report]
    from collections import Counter

    expected = Counter(golden_inputs)
    actual = Counter(actual_inputs)

    if actual != expected:
        missing = list((expected - actual).elements())
        extra = list((actual - expected).elements())

        raise ValueError(
            f"{label} questions do not match approved goldens. "
            f"Missing: {missing}; Extra or duplicate: {extra}"
        )


    metric_buckets = {name: [] for name in EXPECTED_METRICS}
    for i, row in enumerate(report, start=1):
        if not isinstance(row.get("actual_output"), str) or not row["actual_output"].strip():
            raise ValueError(f"{label} case {i}: missing actual_output")
        metrics = row.get("metrics")
        if not isinstance(metrics, list):
            raise ValueError(f"{label} case {i}: missing metrics list")
        seen = set()
        for metric in metrics:
            if not isinstance(metric, dict) or not isinstance(metric.get("name"), str):
                raise ValueError(f"{label} case {i}: invalid metric entry")
            key = metric_key(metric["name"])
            if key in seen:
                raise ValueError(f"{label} case {i}: duplicate metric {key}")
            seen.add(key)
            if key not in EXPECTED_METRICS:
                continue
            score = metric.get("score")
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
                raise ValueError(f"{label} case {i}: invalid score for {key}: {score!r}")
            if not 0 <= score <= 1:
                raise ValueError(f"{label} case {i}: out-of-range score for {key}: {score}")
            if metric.get("error"):
                raise ValueError(f"{label} case {i}: metric {key} errored: {metric['error']}")
            metric_buckets[key].append(float(score))
        missing = EXPECTED_METRICS - seen
        if missing:
            raise ValueError(f"{label} case {i}: missing metrics {sorted(missing)}")

    return {key: sum(scores) / len(scores) for key, scores in metric_buckets.items()}


def make_summary(baseline: dict[str, float], candidate: dict[str, float], tolerance: float,
                 baseline_path: Path, candidate_path: Path, count: int) -> tuple[str, list[str]]:
    lines = [
        "# VeriRAG-Pro DeepEval Regression Report", "",
        f"- Approved golden questions: **{count}** (fixed across both reports)",
        f"- Baseline: `{baseline_path}`",
        f"- Candidate: `{candidate_path}`",
        f"- Allowed drop per metric: **{tolerance:.3f}**", "",
        "| Metric | Baseline | Candidate | Difference | Status |",
        "|---|---:|---:|---:|---|",
    ]
    regressions = []
    for key in sorted(EXPECTED_METRICS):
        delta = candidate[key] - baseline[key]
        failed = delta < -tolerance - 1e-12
        status = "REGRESSION" if failed else "OK"
        lines.append(f"| {key} | {baseline[key]:.4f} | {candidate[key]:.4f} | {delta:+.4f} | {status} |")
        if failed:
            regressions.append(f"{key}: dropped {abs(delta):.4f} (allowed {tolerance:.4f})")
    lines.extend(["", "## Result", ""])
    if regressions:
        lines.append(f"**Regression detected in {len(regressions)} metric(s).**")
        lines.extend(f"- {item}" for item in regressions)
    else:
        lines.append("**No metric dropped beyond the configured tolerance.**")
    lines.extend(["", "This is an offline comparison. It does not run RAG or DeepEval; a fresh candidate report is required to test code changes.", ""])
    return "\n".join(lines), regressions


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline DeepEval regression comparison on fixed goldens")
    parser.add_argument("--goldens", type=Path, default=DEFAULT_GOLDENS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Maximum allowed metric score decrease")
    parser.add_argument("--ci", action="store_true", help="Return exit code 1 when a regression is detected")
    args = parser.parse_args()

    if not math.isfinite(args.tolerance) or not 0 <= args.tolerance <= 1:
        parser.error("--tolerance must be between 0 and 1")

    goldens = read_list(args.goldens, "Golden dataset")
    golden_inputs = [g.get("input") for g in goldens]
    if any(not isinstance(q, str) or not q.strip() for q in golden_inputs):
        raise ValueError("Golden dataset contains an empty or invalid input")
    if len(set(golden_inputs)) != len(golden_inputs):
        raise ValueError("Golden dataset contains duplicate questions")

    baseline_rows = read_list(args.baseline, "Baseline DeepEval report")
    candidate_rows = read_list(args.candidate, "Candidate DeepEval report")
    baseline = validate_and_summarize(baseline_rows, golden_inputs, "Baseline")
    candidate = validate_and_summarize(candidate_rows, golden_inputs, "Candidate")
    summary, regressions = make_summary(
        baseline, candidate, args.tolerance, args.baseline, args.candidate, len(golden_inputs)
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Report saved: {args.report}")

    gh_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as f:
            f.write(summary + "\n")

    if regressions and args.ci:
        print("CI gate failed: regression detected", file=sys.stderr)
        return 1
    if regressions:
        print("Regressions detected (informational mode; use --ci to fail)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Evaluation comparison error: {exc}", file=sys.stderr)
        sys.exit(2)
