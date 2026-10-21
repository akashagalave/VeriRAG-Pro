
from pathlib import Path

import yaml

from config.config_types import Params


ROOT_DIR = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_config(params_path: Path | str = PARAMS_PATH) -> Params:
    """Load and validate experiment parameters."""

    params_path = Path(params_path)

    if not params_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {params_path}"
        )

    with params_path.open("r", encoding="utf-8") as file:
        yaml_config = yaml.safe_load(file)

    if not isinstance(yaml_config, dict):
        raise ValueError(
            "params.yaml must contain a YAML mapping."
        )

    return Params.model_validate(yaml_config, strict=True)


params_config = load_config()


if __name__ == "__main__":
    print("Configuration loaded successfully!")

    print("Chunk size:", params_config.rag_app.chunk_size)
    print("Chunk overlap:", params_config.rag_app.chunk_overlap)
    print("Retrieval k:", params_config.rag_app.k)
    print("Judge model:", params_config.evaluation.judge_llm)

    print(
        "MLflow experiment:",
        params_config.mlflow.experiment_name,
    )
