import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project paths
ROOT_DIR = Path(__file__).resolve().parent

PDF_PATH = (
    ROOT_DIR / "documents" / "Attention_Is_All_You_Need.pdf"
)

OUTPUT_PATH = (
    ROOT_DIR / "data" / "evaluation"
    / "goldens" / "attention_goldens.json"
)

# Golden generation settings
MODEL = "gpt-4.1-mini"
MAX_CONTEXTS = 5
GOLDENS_PER_CONTEXT = 2
MAX_GOLDENS = 10


def generate_goldens():

    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Research paper not found: {PDF_PATH}"
        )

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Golden dataset already exists: {OUTPUT_PATH}. "
            "Move or rename it before generating again."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Initialize DeepEval Synthesizer
    synthesizer = Synthesizer(model=MODEL)

    # Generate goldens from the research paper
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[str(PDF_PATH)],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=MAX_CONTEXTS
        )
    )

    # Keep only valid question-answer pairs
    pairs = [
        {
            "input": golden.input,
            "expected_output": golden.expected_output
        }
        for golden in goldens
        if golden.input and golden.expected_output
    ][:MAX_GOLDENS]

    if not pairs:
        raise RuntimeError(
            "No valid golden Q&A pairs were generated."
        )

    # Save the golden dataset
    OUTPUT_PATH.write_text(
        json.dumps(
            pairs,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    logger.info(
        "Saved %d goldens to %s",
        len(pairs),
        OUTPUT_PATH
    )

    for index, pair in enumerate(pairs, start=1):
        print(f"{index}. {pair['input']}")


if __name__ == "__main__":
    generate_goldens()
