
from pydantic import BaseModel, Field, ConfigDict, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RagAppConfig(StrictConfig):
    prompt_label: str
    llm: str
    embedding_model: str
    embedding_dimensions: int = Field(gt=0)

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    search_type: str
    k: int = Field(ge=1, le=10)

    @model_validator(mode="after")
    def validate_chunking(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )
        return self


class FiltrationConfig(StrictConfig):
    filtration_threshold: float = Field(ge=0, le=1)
    max_quality_retries: int = Field(ge=0)
    filtration_critic_model: str


class EvolutionConfig(StrictConfig):
    num_evolutions: int = Field(ge=0)
    evolutions: dict[str, float]

    @model_validator(mode="after")
    def validate_evolution_weights(self):
        if any(not 0 <= value <= 1 for value in self.evolutions.values()):
            raise ValueError("Evolution weights must be between 0 and 1")

        if abs(sum(self.evolutions.values()) - 1.0) > 1e-6:
            raise ValueError("Evolution weights must sum to 1.0")

        return self


class ContextConstructionConfig(StrictConfig):
    context_critic_model: str
    max_contexts_per_document: int = Field(ge=1)
    context_threshold: float = Field(ge=0, le=1)
    max_context_retries: int = Field(ge=0)
    context_chunk_size: int = Field(gt=0)
    context_chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_context_chunking(self):
        if self.context_chunk_overlap >= self.context_chunk_size:
            raise ValueError(
                "context_chunk_overlap must be smaller than "
                "context_chunk_size"
            )
        return self


class GoldenDatasetConfig(StrictConfig):
    dataset_model: str
    max_goldens_per_context: int = Field(ge=1)
    golden_dataset_filename: str

    filtration_config: FiltrationConfig
    evolution_config: EvolutionConfig
    context_construction_config: ContextConstructionConfig


class EvaluationDatasetConfig(StrictConfig):
    evaluation_dataset_filename: str


class AsyncConfig(StrictConfig):
    throttle_value: int = Field(ge=0)
    max_concurrent: int = Field(ge=1)


class DisplayConfig(StrictConfig):
    results_dir: str


class EvaluationConfig(StrictConfig):
    judge_llm: str
    async_config: AsyncConfig
    display_config: DisplayConfig


class MLflowConfig(StrictConfig):
    tracking_uri: str
    experiment_name: str


class Params(StrictConfig):
    rag_app: RagAppConfig
    golden_dataset: GoldenDatasetConfig
    evaluation_dataset: EvaluationDatasetConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig
