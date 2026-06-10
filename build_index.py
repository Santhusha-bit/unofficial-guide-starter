"""
One-time script: ingest documents → chunk → embed → store in ChromaDB.
Run this before launching app.py.

Usage:
    python build_index.py
"""
from pathlib import Path
from ingest import load_documents, chunk_text
from embed_store import embed_and_store

DOCUMENTS_FOLDER = str(Path(__file__).parent / "documents")


def main() -> None:
    print("Loading documents...")
    docs = load_documents(DOCUMENTS_FOLDER)
    if not docs:
        raise FileNotFoundError(f"No .txt files found in '{DOCUMENTS_FOLDER}'")
    print(f"  Loaded {len(docs)} documents")

    print("\nChunking...")
    all_chunks: list[str] = []
    all_sources: list[str] = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        all_chunks.extend(chunks)
        all_sources.extend([doc["source"]] * len(chunks))
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"\n  Total chunks: {len(all_chunks)}")

    print("\nEmbedding and storing in ChromaDB (this may take a minute)...")
    collection = embed_and_store(all_chunks, all_sources)
    print(f"\nDone. {collection.count()} chunks stored in chroma_db/")


if __name__ == "__main__":
    main()
