
"""VeriRAG-Pro MLflow experiment pipeline.

Reuse existing DeepEval results without running OpenAI.

Usage:
    python execute_experiment_pipeline.py
"""

from pathlib import Path

from config.parameter_config import params_config
from mlflow_tracking import log_experiment


ROOT_DIR = Path(__file__).resolve().parent


def main():

    # Existing 10-question DeepEval baseline
    results_path = (
        ROOT_DIR
        / "reports/evaluation/attention_metrics_k3_baseline.json"
    )

    eval_dataset_path = (
        ROOT_DIR
        / "data/evaluation/eval_dataset/attention_eval_dataset.json"
    )

    golden_dataset_path = (
        ROOT_DIR
        / "data/evaluation/goldens/attention_goldens.json"
    )

    # Historical baseline configuration
    # These must match the settings used to create the saved results.
    experiment_params = {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "retrieval_k": 3,
        "judge_model": "gpt-4.1-mini",
    }

    run_name = "attention_k3_full10_baseline_reuse"

    print("Experiment:", run_name)
    print("Results:", results_path)
    print("Parameters:", experiment_params)

    run_id = log_experiment(
        results_path=results_path,
        run_name=run_name,
        evaluation_dataset_path=eval_dataset_path,
        golden_dataset_path=golden_dataset_path,
        experiment_params=experiment_params,
    )

    print("\nExperiment completed!")
    print("MLflow run ID:", run_id)


if __name__ == "__main__":
    main()
