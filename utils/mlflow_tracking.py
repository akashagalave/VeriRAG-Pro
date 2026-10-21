
import json
import re
from pathlib import Path

import dagshub
import mlflow
from dotenv import load_dotenv

from config.parameter_config import params_config


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent

DEFAULT_RESULTS = (
    ROOT_DIR
    / "reports/evaluation/attention_metrics_k3_baseline.json"
)

DEFAULT_EVAL_DATASET = (
    ROOT_DIR
    / "data/evaluation/eval_dataset/attention_eval_dataset.json"
)

DEFAULT_GOLDENS = (
    ROOT_DIR
    / "data/evaluation/goldens/attention_goldens.json"
)


def metric_key(name: str) -> str:
    return re.sub(
        r"[^a-z0-9]+", "_", name.lower()
    ).strip("_")


def load_evaluation_results(path: Path) -> list:
    with path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, list) or not results:
        raise ValueError(
            "Expected a nonempty list of evaluation cases."
        )

    return results


def calculate_average_metrics(results: list) -> dict:
    grouped = {}

    for case in results:
        for metric in case["metrics"]:
            score = metric.get("score")

            if score is None:
                continue

            key = metric_key(metric["name"])
            grouped.setdefault(key, []).append(float(score))

    return {
        name: sum(scores) / len(scores)
        for name, scores in grouped.items()
    }


def log_experiment(
    results_path: str | Path,
    run_name: str,
    evaluation_dataset_path: str | Path | None = None,
    golden_dataset_path: str | Path | None = None,
    experiment_params: dict | None = None,
):
    """
    Log existing DeepEval results to MLflow.

    Does not run DeepEval, LangGraph, Qdrant or OpenAI.
    """

    results_path = Path(results_path)

    if not results_path.is_file():
        raise FileNotFoundError(results_path)

    results = load_evaluation_results(results_path)
    metrics = calculate_average_metrics(results)

    # Explicit parameters are required to prevent historical
    # experiments being mislabeled with today's YAML values.
    if experiment_params is None:
        raise ValueError(
            "Provide the configuration actually used "
            "to generate this experiment's results."
        )

    required_params = {
        "chunk_size",
        "chunk_overlap",
        "retrieval_k",
        "judge_model",
    }

    missing = required_params - experiment_params.keys()

    if missing:
        raise ValueError(
            f"Missing experiment parameters: {sorted(missing)}"
        )

    print(f"Loaded {len(results)} evaluated case(s)")
    print(f"Found {len(metrics)} metrics")

    # Initialize tracking only when logging is requested.
    dagshub.init(
        repo_owner="akashagalaveaaa1",
        repo_name="VeriRAG-Pro",
        mlflow=True,
    )

    mlflow.set_tracking_uri(
        params_config.mlflow.tracking_uri
    )

    mlflow.set_experiment(
        params_config.mlflow.experiment_name
    )

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_params({
            **experiment_params,
            "evaluated_cases": len(results),
        })

        mlflow.log_metrics(metrics)

        mlflow.set_tags({
            "evaluation_type": (
                "smoke_test"
                if len(results) == 1
                else "full_evaluation"
            ),
            "dataset": "Attention Is All You Need",
            "evaluation_framework": "DeepEval",
            "experiment_purpose": "manual_rag_optimization",
            "pdf_parser": "pymupdf",
            "chunking_strategy": "recursive_character",
        })

        mlflow.log_artifact(
            str(results_path),
            artifact_path="evaluation_results",
        )

        if evaluation_dataset_path is not None:
            dataset_path = Path(evaluation_dataset_path)

            if not dataset_path.is_file():
                raise FileNotFoundError(dataset_path)

            mlflow.log_artifact(
                str(dataset_path),
                artifact_path="evaluation_dataset",
            )

        if golden_dataset_path is not None:
            golden_path = Path(golden_dataset_path)

            if not golden_path.is_file():
                raise FileNotFoundError(golden_path)

            mlflow.log_artifact(
                str(golden_path),
                artifact_path="golden_dataset",
            )

        print("\nMetrics logged:")

        for name, score in metrics.items():
            print(f"  {name}: {score:.4f}")

        print(f"\nMLflow run ID: {run.info.run_id}")
        print("MLflow tracking completed successfully!")

        return run.info.run_id


if __name__ == "__main__":
    print(
        "Use execute_experiment_pipeline.py "
        "to log an experiment."
    )
