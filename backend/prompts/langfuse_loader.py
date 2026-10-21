import logging
import os

from dotenv import load_dotenv
from langfuse import get_client

from backend.prompts.registry import PROMPT_LABEL, PROMPT_TEXTS

load_dotenv()

logger = logging.getLogger(__name__)


def load_prompt(name: str):
    """Fetch a prompt from Langfuse or use its local fallback.

    Returns a Langfuse prompt object supporting .compile().
    """

    if name not in PROMPT_TEXTS:
        raise ValueError(f"Unknown RAG prompt: {name}")

    label = os.getenv("LANGFUSE_PROMPT_LABEL", PROMPT_LABEL)

    langfuse = get_client()

    prompt = langfuse.get_prompt(
        name=name,
        type="text",
        label=label,
        fallback=PROMPT_TEXTS[name],
    )

    if prompt.is_fallback:
        logger.warning(
            "Langfuse unavailable or prompt missing: %s. "
            "Using local fallback.",
            name,
        )
    else:
        logger.info(
            "Loaded Langfuse prompt '%s', version %s",
            name,
            prompt.version,
        )

    return prompt
