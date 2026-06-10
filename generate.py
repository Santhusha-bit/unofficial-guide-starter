"""
Milestone 5 - Answer generation via Groq.
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You are a knowledgeable guide for AI/ML research onboarding. You help students and \
self-learners navigate the AI/ML research world using the provided context.

Rules:
- Answer based primarily on the provided context.
- Be specific and practical - avoid vague generic advice.
- Cite sources by filename when you draw from them.
- If the context does not contain enough information to answer, say so clearly rather \
than making things up.\
"""


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """Call Groq with the retrieved context and return the model's answer."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    from embed_store import get_collection, retrieve

    collection = get_collection()
    query = "What is the three-pass method for reading a research paper?"
    chunks = retrieve(query, collection, k=5)
    answer = generate_answer(query, chunks)
    print(f"Q: {query}\n\nA: {answer}")
