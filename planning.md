# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

**AI/ML Research Onboarding** — an unofficial guide for students and self-learners trying to break into AI/ML research.

Official university syllabi and course pages tell you *what* to study, but they don't tell you how to actually navigate the research world: how to read a paper efficiently, how to find and cold-email a research advisor, which resources practitioners actually recommend, or how researchers stay current with the literature. This tribal knowledge lives scattered across Reddit threads, personal blogs, and community forums — exactly the kind of content a RAG system is well-suited to surface.

---

## Documents

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | r/MachineLearning wiki & top posts | Community-curated resource lists and FAQs from the main ML subreddit | https://www.reddit.com/r/MachineLearning/wiki/index |
| 2 | r/learnmachinelearning FAQ | Beginner-focused advice threads: where to start, math prerequisites, project ideas | https://www.reddit.com/r/learnmachinelearning/wiki/faq |
| 3 | Andrej Karpathy — "A Survival Guide to a PhD" | Practitioner advice on picking advisors, reading papers, and building research taste | https://karpathy.github.io/2016/09/07/phd/ |
| 4 | Chip Huyen — ML career and interviews blog | Advice on ML roles, research vs. engineering, interview preparation | https://huyenchip.com/blog/ |
| 5 | "How to Read a Paper" by S. Keshav | Classic three-pass method for efficiently reading research papers | documents/keshav_how_to_read_a_paper.pdf |
| 6 | Jay Alammar — "The Illustrated Transformer" | Visual, intuitive explanation of the transformer architecture | https://jalammar.github.io/illustrated-transformer/ |
| 7 | Distill.pub — "Attention and Augmented RNNs" | Interactive article explaining attention mechanisms with visuals | https://distill.pub/2016/augmented-rnns/ |
| 8 | Sebastian Ruder — "An Overview of Gradient Descent Optimization Algorithms" | Deep-dive blog post on optimization methods referenced constantly in ML research | https://www.ruder.io/optimizing-gradient-descent/ |
| 9 | fast.ai forums — "How do I get into ML research?" thread | Community discussion with practitioner responses on transitioning into research | https://forums.fast.ai |
| 10 | ML Collective — "Open Questions in ML" and onboarding resources | Community docs for independent researchers with no lab affiliation | https://mlcollective.org/resources/ |
| 11 | Papers With Code — Methods pages | Structured descriptions of common ML methods with linked papers and benchmarks | https://paperswithcode.com/methods |
| 12 | r/MachineLearning "What papers should I read first?" megathread | Community-ranked list of foundational and landmark papers | https://www.reddit.com/r/MachineLearning/ |

---

## Chunking Strategy

**Chunk size:** 512 tokens

**Overlap:** 64 tokens

**Reasoning:** The corpus mixes blog posts, forum threads, and short PDF excerpts — all paragraph-driven content where a single idea typically fits in 2–4 sentences. 512 tokens comfortably holds a full argument or explanation without splitting mid-thought. A 64-token overlap (≈ 1–2 sentences) preserves enough context so that a chunk retrieved in isolation still makes sense, especially for list-style content (e.g., "Step 3: …") where the preceding steps matter. Going larger (e.g., 1024 tokens) risks mixing multiple unrelated ideas into one chunk, hurting retrieval precision on narrow questions like "what is the three-pass method?"

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`

**Top-k:** 5

**Production tradeoff reflection:** `all-MiniLM-L6-v2` is fast and lightweight — ideal for a class project — but it was trained on general web text, not ML research prose. In a real deployment I would weigh the following tradeoffs:

- **Domain accuracy:** A science-specific model like `allenai-specter` (trained on paper citations) would embed ML terminology more faithfully. Worth the accuracy gain if the corpus is heavy on paper abstracts.
- **Context length:** `all-MiniLM-L6-v2` caps at 256 tokens; for longer chunks, `all-mpnet-base-v2` (512 tokens) avoids truncation silently corrupting embeddings.
- **Multilingual support:** Not needed here, but `paraphrase-multilingual-MiniLM-L12-v2` would matter for a global research community use case.
- **Latency vs. accuracy:** OpenAI's `text-embedding-3-large` consistently outperforms open models on retrieval benchmarks but adds API cost and latency per query — acceptable for low-traffic tools, problematic at scale.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What is the three-pass method for reading a research paper? | Keshav's three-pass method: first pass (5 min, titles/abstract/conclusions), second pass (1 hr, figures and references), third pass (4 hrs, full reconstruction). Source: Keshav PDF. |
| 2 | How should I cold-email a professor to ask about research opportunities? | Keep it short, show you've read their work, mention a specific paper and why it interests you, attach a resume. Source: Karpathy blog and r/learnmachinelearning threads. |
| 3 | What math do I need before reading transformer papers? | Linear algebra (matrix multiplication, eigendecomposition), probability/statistics (Bayes' rule, distributions), and calculus (chain rule for backprop). Sources: r/learnmachinelearning FAQ, Ruder blog. |
| 4 | How do ML researchers stay current with new papers? | ArXiv daily digest, Twitter/X following key researchers, Papers With Code alerts, ML newsletter subscriptions (e.g., The Batch, Import AI). Source: r/MachineLearning threads. |
| 5 | What is the difference between a research engineer and a research scientist role at a tech company? | Research engineers implement and scale ideas; research scientists generate novel ideas and publish. Engineers need stronger software skills; scientists need stronger publication records. Source: Chip Huyen blog. |

---

## Anticipated Challenges

1. **Noisy, off-topic forum content:** Reddit threads and forum posts contain tangents, jokes, and outdated advice mixed with genuinely useful responses. A chunk might retrieve a highly-upvoted comment that's actually sarcastic or irrelevant to the query. Mitigation: manually review and filter the scraped forum content before ingestion; prefer top-level posts and highly-upvoted comments over full thread dumps.

2. **Inconsistent terminology across sources:** Different sources use different terms for the same concept (e.g., "attention head" vs. "self-attention mechanism" vs. "multi-head attention"). The embedding model may fail to link these as semantically equivalent, causing relevant chunks to score below the top-k threshold. Mitigation: use a slightly higher top-k (e.g., 7) during evaluation and assess whether paraphrased queries surface the same answers.

---

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────────────┐
│  Document Ingestion  │────▶│      Chunking         │────▶│  Embedding + Vector Store  │
│                     │     │                      │     │                            │
│  • requests/        │     │  • chunk_text()      │     │  • all-MiniLM-L6-v2        │
│    BeautifulSoup    │     │  • 512 token chunks  │     │    (sentence-transformers) │
│    (HTML scraping)  │     │  • 64 token overlap  │     │  • ChromaDB                │
│  • pdfplumber       │     │  • plain .txt output │     │    (local persistent store)│
│    (PDF extraction) │     │                      │     │                            │
└─────────────────────┘     └──────────────────────┘     └────────────────┬───────────┘
                                                                           │
                                                          ┌────────────────▼───────────┐
                                                          │        Retrieval            │
                                                          │                            │
                                                          │  • query → embed           │
                                                          │  • ChromaDB similarity     │
                                                          │    search (top-k = 5)      │
                                                          └────────────────┬───────────┘
                                                                           │
                                                          ┌────────────────▼───────────┐
                                                          │        Generation           │
                                                          │                            │
                                                          │  • Groq API                │
                                                          │    (llama-3.3-70b-versatile)│
                                                          │  • Gradio chat interface   │
                                                          └────────────────────────────┘
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**

I'll give Claude the **Chunking Strategy** section and the **Documents** table and ask it to implement two functions: `ingest_documents(folder_path) -> list[str]` (reads `.txt` and `.pdf` files from the `documents/` folder) and `chunk_text(text, chunk_size=512, overlap=64) -> list[str]` (splits on token boundaries using a sliding window). I'll verify by printing the length and first 3 chunks of one ingested document and confirming chunk sizes are within ±10% of 512 tokens.

**Milestone 4 — Embedding and retrieval:**

I'll give Claude the **Retrieval Approach** section and ask it to implement `embed_and_store(chunks, collection_name) -> ChromaDB collection` using `sentence-transformers` and `chromadb`, and `retrieve(query, collection, k=5) -> list[str]`. I'll verify by running each of my 5 evaluation questions as retrieval queries and manually checking that at least 3 of the 5 returned chunks are relevant to the question.

**Milestone 5 — Generation and interface:**

I'll give Claude the full `planning.md` plus the working retrieval output from Milestone 4 and ask it to implement `generate_answer(query, retrieved_chunks) -> str` using the Groq API with `llama-3.3-70b-versatile`, and wrap it in a Gradio chat interface. I'll verify by running all 5 evaluation questions end-to-end and scoring each response against the expected answers in the Evaluation Plan table.
