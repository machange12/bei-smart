"""ChromaDB vector store over the AgriPulse knowledge base.

Used only for the `explain` intent. Numeric questions (forecasts, alerts,
comparisons) never touch this module -- they are answered by lookup.py's
deterministic dataframe filters instead, because forecast rows embed too
similarly to each other for semantic search to reliably pick the right one.
"""

from __future__ import annotations

from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "kb"
CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chroma"
COLLECTION_NAME = "agripulse_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_client = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _get_embedding_function():
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _load_kb_documents() -> tuple[list[str], list[str], list[dict]]:
    ids, documents, metadatas = [], [], []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        ids.append(path.stem)
        documents.append(text)
        metadatas.append({"source": path.name, "title": title})
    return ids, documents, metadatas


def build_index(force: bool = False) -> int:
    """(Re)build the Chroma collection from app/data/kb/*.md. Returns the
    number of documents indexed."""
    client = _get_client()
    ef = _get_embedding_function()

    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

    ids, documents, metadatas = _load_kb_documents()
    if not ids:
        return 0

    if collection.count() == 0 or force:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return len(ids)


def get_collection():
    global _collection
    if _collection is None:
        client = _get_client()
        ef = _get_embedding_function()
        collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
        if collection.count() == 0:
            build_index(force=True)
            collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
        _collection = collection
    return _collection


def query(question: str, n_results: int = 3) -> list[dict]:
    """Return the top-n KB chunks for an explanatory question, each with
    its source filename, title, text and a distance score (lower = closer)."""
    collection = get_collection()
    result = collection.query(query_texts=[question], n_results=n_results)

    hits = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for doc, meta, distance in zip(docs, metas, distances):
        hits.append(
            {
                "source": meta.get("source"),
                "title": meta.get("title"),
                "text": doc,
                "distance": distance,
            }
        )
    return hits
