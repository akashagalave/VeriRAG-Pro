from dotenv import load_dotenv
from langfuse import get_client

from backend.prompts.registry import (
    PROMPT_LABEL,
    PROMPT_TEXTS,
)

load_dotenv()


def is_up_to_date(langfuse, name: str, text: str) -> bool:
    """Check whether the latest Langfuse prompt matches local text."""
    try:
        existing = langfuse.get_prompt(
            name=name,
            type="text",
            label=PROMPT_LABEL,
            cache_ttl_seconds=0,
        )
        return existing.prompt == text
    except Exception:
        return False


def push_prompts():
    langfuse = get_client()

    for name, text in PROMPT_TEXTS.items():
        if is_up_to_date(langfuse, name, text):
            print(f"{name}: unchanged, skipped")
            continue

        created = langfuse.create_prompt(
            name=name,
            type="text",
            prompt=text,
        )

        print(
            f"{name}: created version {created.version}"
        )

    langfuse.flush()


if __name__ == "__main__":
    push_prompts()