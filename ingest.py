"""
Milestone 3 - Document ingestion and chunking.
"""
from pathlib import Path


def load_documents(folder_path: str) -> list[dict]:
    """Read every .txt file in folder_path; return list of {source, text} dicts."""
    folder = Path(folder_path)
    docs = []
    for path in sorted(folder.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": path.name, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split text into overlapping chunks.

    chunk_size and overlap are in tokens; uses the approximation 1 token ≈ 0.75 words.
    Slides a window of chunk_words over the word list with a stride of
    (chunk_words - overlap_words), so adjacent chunks share overlap_words words.
    """
    chunk_words = int(chunk_size * 0.75)   # ≈ 384 words
    overlap_words = int(overlap * 0.75)     # ≈ 48 words
    stride = chunk_words - overlap_words

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride

    return chunks


if __name__ == "__main__":
    docs = load_documents("documents")
    print(f"Loaded {len(docs)} documents")

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"\nFirst chunk preview:\n{all_chunks[0][:300]}...")
