"""
Milestone 4 — Embedding and vector store (ChromaDB).
"""
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ml_research_guide"
CHROMA_PATH = str(Path(__file__).parent / "chroma_db")

# Module-level model cache so we don't reload on every call
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_and_store(
    chunks: list[str],
    sources: list[str],
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Embed chunks with all-MiniLM-L6-v2 and persist them in ChromaDB."""
    model = _get_model()
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop and recreate so re-runs start fresh
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_sources = sources[i : i + batch_size]
        embeddings = model.encode(batch_chunks, show_progress_bar=True)
        collection.add(
            documents=batch_chunks,
            embeddings=embeddings.tolist(),
            metadatas=[{"source": s} for s in batch_sources],
            ids=[f"chunk_{i + j}" for j in range(len(batch_chunks))],
        )

    return collection


def get_collection(collection_name: str = COLLECTION_NAME) -> chromadb.Collection:
    """Load an existing ChromaDB collection (call after build_index.py has run)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(collection_name)


def retrieve(
    query: str,
    collection: chromadb.Collection,
    k: int = 5,
) -> list[dict]:
    """Return the top-k chunks most semantically similar to query."""
    model = _get_model()
    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {"text": doc, "source": meta["source"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


if __name__ == "__main__":
    collection = get_collection()
    print(f"Collection has {collection.count()} chunks")

    test_query = "What is the three-pass method for reading a research paper?"
    results = retrieve(test_query, collection)
    print(f"\nTop {len(results)} results for: '{test_query}'\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['source']} (distance={r['distance']:.4f})")
        print(f"    {r['text'][:200]}...\n")
