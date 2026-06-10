"""
Milestone 5 - Gradio chat interface for the ML Research Onboarding Guide.

Usage:
    1. python build_index.py   (first time only)
    2. python app.py
"""
import os
import gradio as gr
from dotenv import load_dotenv
from embed_store import get_collection, retrieve
from generate import generate_answer

load_dotenv()

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


def chat(message: str, history: list) -> str:
    collection = _get_collection()
    chunks = retrieve(message, collection, k=5)
    answer = generate_answer(message, chunks)

    # Append deduplicated source list
    seen = []
    for c in chunks:
        if c["source"] not in seen:
            seen.append(c["source"])
    sources_line = "\n\n---\n*Sources: " + ", ".join(seen) + "*"

    return answer + sources_line


demo = gr.ChatInterface(
    fn=chat,
    title="The Unofficial ML Research Guide",
    description=(
        "Ask anything about breaking into AI/ML research - reading papers, "
        "understanding architectures, finding advisors, or navigating the research world."
    ),
    examples=[
        "What is the three-pass method for reading a research paper?",
        "How should I cold-email a professor to ask about research opportunities?",
        "What math do I need before reading transformer papers?",
        "How do ML researchers stay current with new papers?",
        "What is the difference between a research engineer and a research scientist?",
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY not set. Add it to a .env file.")
    demo.launch()
