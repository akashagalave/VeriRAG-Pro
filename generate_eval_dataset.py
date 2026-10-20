"""Generate VeriRAG-Pro evaluation cases from approved Attention goldens.

Usage:
    python generate_eval_dataset.py --limit 1   # inexpensive smoke test
    python generate_eval_dataset.py             # all 10 goldens

This script does not touch evaluate.py, eval_baseline.json, or CI/CD.
"""

import argparse
import json
import logging
import sqlite3
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from backend.paper_loader import load_pdf
from config.parameter_config import params_config

load_dotenv()

from backend.rag_graph import build_graph
from backend.vector_store import add_paper, get_collection_name, qdrant_client

logger = logging.getLogger("attention_eval_dataset")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "documents" / "Attention_Is_All_You_Need.pdf"
GOLDENS_PATH = ROOT / "data" / "evaluation" / "goldens" / "attention_goldens.json"
OUTPUT_DIR = ROOT / "data" / "evaluation" / "eval_dataset"
MAX_GOLDENS = 10

# Same retrieval nudge used in existing evaluate.py, so we exercise paper retrieval.
# Keep the original question in the stored 'input' field for later metric evaluation.
RETRIEVAL_SUFFIX = " as per the report in knowledge base"


def run_one(graph, question: str, session_id: str, thread_id: str) -> dict:
    asked_question = question + RETRIEVAL_SUFFIX
    state = {
        "messages": [HumanMessage(content=asked_question)],
        "session_id": session_id,
        "query": asked_question,
        "route": None,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "claim_verdict": None,
        "claim_source": None,
        "superseding_papers": [],
        "answer": None,
        "is_relevant": None,
        "rewrite_count": 0,
    }
    final = graph.invoke(state, config={
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 30,
    })
    docs = final.get("retrieved_docs") or []
    return {
        "input": question,
        "actual_output": final.get("answer") or "",
        "retrieval_context": [doc.page_content for doc in docs],
        "route": final.get("route") or "unknown",
        "asked_query": asked_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an Attention RAG evaluation dataset")
    parser.add_argument("--limit", type=int, default=MAX_GOLDENS,
                        help="Number of goldens to execute, 1..10 (default 10)")
    args = parser.parse_args()
    if not 1 <= args.limit <= MAX_GOLDENS:
        parser.error("--limit must be between 1 and 10")

    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Research PDF missing: {PDF_PATH}")
    if not GOLDENS_PATH.is_file():
        raise FileNotFoundError(f"Goldens missing: {GOLDENS_PATH}")

    goldens = json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))
    if not isinstance(goldens, list) or len(goldens) < args.limit:
        raise ValueError(f"Expected at least {args.limit} golden pairs")
    selected = goldens[:args.limit]
    for i, pair in enumerate(selected, 1):
        if not pair.get("input") or not pair.get("expected_output"):
            raise ValueError(f"Golden {i} is missing input or expected_output")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / (
        "attention_eval_dataset_smoke.json" if args.limit == 1
        else "attention_eval_dataset.json" if args.limit == MAX_GOLDENS
        else f"attention_eval_dataset_{args.limit}.json"
    )
    if output.exists():
        raise FileExistsError(f"Output already exists; move it before rerunning: {output}")

    # Unique IDs avoid deleting/overwriting the old project's eval collection.
    session_id = f"attention_eval_{uuid.uuid4().hex[:12]}"
    collection = get_collection_name(session_id)
    # A separate temporary local checkpoint file per run; not the app's checkpoint DB.
    checkpoint_file = OUTPUT_DIR / f".{session_id}_checkpoints.db"
    conn = None
    collection_created = False
    try:
        logger.info("Loading and indexing paper once in collection %s", collection)
        docs = load_pdf(
                str(PDF_PATH),
                chunk_size=params_config.rag_app.chunk_size,
                chunk_overlap=params_config.rag_app.chunk_overlap,
                    )

        logger.info(
                    "Experiment chunking: size=%d, overlap=%d",
                    params_config.rag_app.chunk_size,
                    params_config.rag_app.chunk_overlap,
                        )

        add_paper(docs, session_id)
        collection_created = True
        logger.info("Indexed %d document chunks", len(docs))

        conn = sqlite3.connect(str(checkpoint_file), check_same_thread=False)
        graph = build_graph(checkpointer=SqliteSaver(conn))
        rows = []
        for idx, golden in enumerate(selected, start=1):
            logger.info("Running golden %d/%d: %s", idx, len(selected), golden["input"])
            result = run_one(graph, golden["input"], session_id,
                             f"{session_id}_golden_{idx:03d}")
            result["expected_output"] = golden["expected_output"]
            rows.append(result)
            logger.info("route=%s | retrieved chunks=%d | answer chars=%d",
                        result["route"], len(result["retrieval_context"]),
                        len(result["actual_output"]))
            # Persist completed cases even if a later request fails.
            output.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                              encoding="utf-8")

        print(f"\nSaved {len(rows)} evaluation cases: {output}")
        if any(r["route"] != "retrieve" or not r["retrieval_context"] for r in rows):
            print("WARNING: Some cases did not retrieve context. Inspect routes before scoring RAG metrics.")
    finally:
        if conn is not None:
            conn.close()
        for path in (checkpoint_file, Path(str(checkpoint_file) + "-wal"),
                     Path(str(checkpoint_file) + "-shm")):
            path.unlink(missing_ok=True)
        if collection_created:
            try:
                qdrant_client.delete_collection(collection)
                logger.info("Cleaned up temporary Qdrant collection %s", collection)
            except Exception as exc:
                logger.warning("Could not clean up %s: %s", collection, exc)


if __name__ == "__main__":
    main()
