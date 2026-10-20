
import logging
import os

from langfuse import get_client

logger = logging.getLogger(__name__)


def initialize_langfuse():
    """Initialize Langfuse without blocking API startup."""
    if not (
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    ):
        logger.warning(
            "Langfuse credentials missing; tracing disabled."
        )
        return None

    try:
        client = get_client()

        if client.auth_check():
            logger.info(
                "Langfuse connected successfully (environment=%s)",
                os.getenv(
                    "LANGFUSE_TRACING_ENVIRONMENT",
                    "default",
                ),
            )
        else:
            logger.warning(
                "Langfuse authentication failed; check credentials."
            )
            return None

        return client

    except Exception:
        logger.exception(
            "Langfuse initialization failed; continuing without tracing."
        )
        return None


def shutdown_langfuse(client):
    """Flush buffered tracing data when FastAPI shuts down."""
    if client is None:
        return

    try:
        client.shutdown()
        logger.info("Langfuse shutdown complete.")
    except Exception:
        logger.exception("Failed to shut down Langfuse cleanly.")
