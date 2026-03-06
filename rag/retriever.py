"""
RAG Retriever.
Queries the ChromaDB vector store for relevant math knowledge chunks.
Returns structured results; never hallucinates sources.
"""

import os
from utils.config import CHROMA_PERSIST_DIR, RAG_TOP_K
from utils.logger import get_logger

logger = get_logger("rag.retriever")


def _get_collection():
    """Get the persistent ChromaDB collection."""
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    abs_persist = os.path.abspath(CHROMA_PERSIST_DIR)
    client = chromadb.PersistentClient(path=abs_persist)
    embedding_fn = DefaultEmbeddingFunction()

    try:
        collection = client.get_collection(
            name="math_knowledge",
            embedding_function=embedding_fn,
        )
        return collection
    except Exception as e:
        logger.error("Collection 'math_knowledge' not found. Run ingest first. %s", e)
        return None


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Retrieve the top-k most relevant knowledge chunks for a query.

    Args:
        query: The math problem or topic to search for.
        top_k: Number of results (defaults to config RAG_TOP_K).

    Returns:
        List of dicts with keys: document, source, topic, score.
        Returns empty list if no collection or no results — never hallucinated.
    """
    if top_k is None:
        top_k = RAG_TOP_K

    collection = _get_collection()
    if collection is None:
        logger.warning("No collection available. Returning empty results.")
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
        )
    except Exception as e:
        logger.error("Retrieval query failed: %s", e)
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        logger.info("No relevant documents found for query: %s", query[:80])
        return []

    retrieved = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved.append(
            {
                "document": doc,
                "source": meta.get("source", "unknown"),
                "topic": meta.get("topic", "unknown"),
                "score": round(1.0 / (1.0 + dist), 3),  # Convert distance to score
            }
        )

    logger.info(
        "Retrieved %d docs for query '%s' (top score=%.3f)",
        len(retrieved),
        query[:50],
        retrieved[0]["score"] if retrieved else 0,
    )

    return retrieved


def format_context(results: list[dict]) -> str:
    """
    Format retrieved results into a string context for the LLM.

    Args:
        results: List of retrieval results from retrieve().

    Returns:
        Formatted string with source citations.
    """
    if not results:
        return "No relevant knowledge base documents found."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Source {i}: {r['source']} (score: {r['score']})]\n{r['document']}"
        )
    return "\n\n---\n\n".join(parts)
