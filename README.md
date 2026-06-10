# The Unofficial Guide - Project 1

---

## Domain

**AI/ML Research Onboarding** - a RAG system that answers questions about breaking into AI/ML research.

Official university syllabi and course pages tell you *what* to study, but they don't tell you how to actually navigate the research world: how to read a paper efficiently, how to cold-email a professor, which resources practitioners actually use to stay current, or what separates a research engineer from a research scientist. This tribal knowledge lives scattered across personal blogs, Reddit threads, course notes, and community forums - exactly the kind of content that is valuable precisely because it is not indexed under any single official source. The goal of this system is to surface it through a conversational interface.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Andrej Karpathy - "A Survival Guide to a PhD" | Blog post | https://karpathy.github.io/2016/09/07/phd/ |
| 2 | Jay Alammar - "The Illustrated Transformer" | Blog post | https://jalammar.github.io/illustrated-transformer/ |
| 3 | Jay Alammar - "The Illustrated BERT" | Blog post | https://jalammar.github.io/illustrated-bert/ |
| 4 | Sebastian Ruder - "An Overview of Gradient Descent Optimization Algorithms" | Blog post | https://www.ruder.io/optimizing-gradient-descent/ |
| 5 | Chip Huyen - "Career Advice for Recent CS Graduates" | Blog post | https://huyenchip.com/2018/10/08/career-advice-recent-cs-graduates.html |
| 6 | Chip Huyen - "What I Learned from Looking at 200 ML Tools" | Blog post | https://huyenchip.com/2020/06/22/mlops.html |
| 7 | Distill.pub - "Attention and Augmented Recurrent Neural Networks" | Research article | https://distill.pub/2016/augmented-rnns/ |
| 8 | Stanford CS231n - "Transfer Learning" | Course notes | https://cs231n.github.io/transfer-learning/ |
| 9 | Stanford CS231n - "Convolutional Neural Networks" | Course notes | https://cs231n.github.io/convolutional-networks/ |
| 10 | ML-Ops.org - "MLOps Principles" | Community docs | https://ml-ops.org/content/mlops-principles |
| 11 | How to Read ML Research Papers | Compiled community guide | `documents/how_to_read_ml_papers.txt` |
| 12 | Getting Into ML Research as an Undergrad | Compiled community guide | `documents/getting_into_ml_research.txt` |

---

## Chunking Strategy

**Chunk size:** 512 tokens (implemented as ~384 words using the approximation 1 token ≈ 0.75 words)

**Overlap:** 64 tokens (~48 words)

**Why these choices fit your documents:** The corpus is made up of blog posts, course notes, and community guides - all paragraph-driven prose where a single idea typically fits in 2–5 sentences. A 384-word chunk comfortably holds a complete argument or explanation without splitting it mid-thought. The 48-word overlap means adjacent chunks share roughly one paragraph of context, which preserves coherence for list-style content (e.g., "Step 3 of the three-pass method…") where a chunk retrieved in isolation still needs a sentence or two of preceding context to make sense. The implementation uses a sliding window over the word list rather than a naive fixed split, so chunk boundaries fall between words rather than mid-word.

**Final chunk count:** 43 chunks across 12 documents (3–7 chunks per document).

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`

**Production tradeoff reflection:** `all-MiniLM-L6-v2` is fast, runs locally with no API cost, and produces good general-purpose semantic embeddings - the right choice for a class project. In a real deployment I would weigh several tradeoffs. First, **context length**: this model caps at 256 tokens, so chunks longer than ~192 words are silently truncated during embedding, which can corrupt the representation of the tail of each chunk; `all-mpnet-base-v2` doubles that limit to 512 tokens. Second, **domain accuracy**: both models were trained on general web text, not ML research prose; a science-specific model like `allenai-specter` (trained on paper citation graphs) would embed ML jargon and author-attribution patterns more faithfully. Third, **latency**: local inference adds ~50–100ms per query on CPU, which is acceptable for a single-user demo but would require batching or a GPU for production traffic. Fourth, **multilingual support**: not relevant here, but `paraphrase-multilingual-MiniLM-L12-v2` would matter if the corpus were extended to non-English sources. The OpenAI `text-embedding-3-large` model consistently outperforms open alternatives on retrieval benchmarks but introduces API cost and a hard network dependency.

---

## Grounded Generation

**System prompt grounding instruction:**

```
You are a knowledgeable guide for AI/ML research onboarding. You help students and
self-learners navigate the AI/ML research world using the provided context.

Rules:
- Answer based primarily on the provided context.
- Be specific and practical - avoid vague generic advice.
- Cite sources by filename when you draw from them.
- If the context does not contain enough information to answer, say so clearly rather
  than making things up.
```

The key mechanism is that the user message is structured as `Context:\n[retrieved chunks]\n\nQuestion: [query]` - the model sees the source material inline before the question, so it answers by synthesizing the provided text rather than drawing on parametric memory. The instruction "say so clearly rather than making things up" is the explicit anti-hallucination directive.

**How source attribution is surfaced in the response:** The model is instructed to cite sources by filename within its answer (e.g., `[Source: how_to_read_ml_papers.txt]`). Additionally, `app.py` appends a deduplicated list of the retrieved filenames as a footer to every response, so the user always sees which documents were consulted regardless of whether the model cited them inline.

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What is the three-pass method for reading a research paper? | Keshav's three passes: 5-min bird's-eye view, 1-hr details, 4-hr re-implementation. Five Cs after Pass 1. | Correctly described all three passes with time estimates, the five Cs (Category, Context, Correctness, Contributions, Clarity), and the ML-specific adaptations for each pass. | Relevant | Accurate |
| 2 | How should I cold-email a professor to ask about research opportunities? | Specific subject line, mention a paper you read, 3–4 paragraphs, attach CV, follow up after 2 weeks. | Gave a numbered checklist matching all expected points: specific subject line, genuine paper mention, brief background, stated ask, 3–4 paragraph limit, customise per professor, one follow-up. | Relevant | Accurate |
| 3 | What math do I need before reading transformer papers? | Linear algebra, probability/stats, calculus (chain rule), some information theory. | Correctly listed all four areas with specific sub-topics (eigendecomposition, Bayes' rule, chain rule, KL divergence). Also retrieved a chunk from `illustrated_transformer.txt` describing transformer math notation, which was adjacent but not directly answering "prerequisites." Model handled it by noting that source describes how the math is used, not what to learn first. | Partially relevant (one off-target chunk) | Accurate |
| 4 | How do ML researchers stay current with new papers? | Twitter/X, newsletters (The Batch, Import AI), ArXiv Sanity, journal clubs, lab blogs. | Covered all five expected channels with specific newsletter names and tool names. Also mentioned conference proceedings calendar notifications, which was a useful addition not in the expected answer. | Relevant | Accurate |
| 5 | What is the difference between a research engineer and a research scientist role at a tech company? | Scientists generate ideas and publish; engineers implement and scale. Scientists need publication records; engineers need stronger software skills. | Correctly distinguished the two roles on all expected axes. Added that lines blur at smaller organizations, which is accurate and useful context. | Relevant | Accurate |

---

## Failure Case Analysis

**Question that failed:** "What math do I need before reading transformer papers?" (Q3 - partial retrieval failure)

**What the system returned:** The answer itself was accurate, but one of the five retrieved chunks came from `illustrated_transformer.txt` - a chunk that describes *how* matrix operations are used inside a transformer (QKV matrices, softmax, dot products) rather than *what prerequisites to learn*. The model correctly noted that the source "does not explicitly state the required math prerequisites," but it still spent part of its response paraphrasing that chunk, slightly diluting the answer's focus.

**Root cause (tied to a specific pipeline stage):** This is a retrieval stage failure caused by vocabulary overlap rather than semantic alignment. The query "what math do I need" activates embeddings associated with mathematical concepts. The `illustrated_transformer.txt` chunk contains dense mathematical vocabulary ("matrix multiplication," "softmax," "√d_k," "Q, K, V vectors") that is semantically close to "math for transformers" in the embedding space - even though the chunk is explaining how the architecture works, not listing learning prerequisites. The embedding model has no way to distinguish "math *used in* a transformer" from "math *needed to understand* a transformer" because both involve the same vocabulary.

**What you would change to fix it:** Two options. First, increase top-k from 5 to 7 and add a post-retrieval re-ranking step (e.g., a cross-encoder like `cross-encoder/ms-marco-MiniLM-L-6-v2`) that scores each chunk against the query at full attention, not just cosine similarity - cross-encoders are much better at distinguishing "prerequisite" intent from "concept explanation" intent. Second, add a metadata filter at retrieval time: tag document chunks with their primary topic (e.g., `topic: architecture_explanation` vs `topic: learning_resources`) and narrow the query to only prerequisite-tagged chunks when the question contains words like "need," "before," or "prerequisites."

---

## Spec Reflection

**One way the spec helped you during implementation:**

Having the chunking strategy written out before coding `chunk_text()` made the implementation decision obvious rather than arbitrary. The spec stated "512 tokens, 64 overlap, paragraph-driven prose" which directly translated into the sliding window parameters (`chunk_words = int(512 * 0.75)`, `overlap_words = int(64 * 0.75)`) and justified using a word-split approach rather than a character split. Without the spec, those numbers would have been guesses; with it, they were justified constraints. The same was true for the top-k value - having already decided 5 in planning meant the retrieval call `retrieve(query, collection, k=5)` required no in-the-moment deliberation.

**One way your implementation diverged from the spec, and why:**

The spec listed `documents/keshav_how_to_read_a_paper.pdf` as source #5 (a local PDF). During implementation it became clear that a well-structured `.txt` file synthesising the three-pass method from multiple community sources (Keshav + ML-specific adaptations from practitioner blogs) would be more useful to the RAG system than a raw PDF extraction of the original paper - PDFs often produce garbled text with broken line breaks and citation noise that hurts embedding quality. The PDF source was replaced with `how_to_read_ml_papers.txt`, a compiled guide that covers the same material plus ML-specific adaptations. The planning.md note "Update the Documents section if you change your approach" was followed accordingly.

---

## AI Usage

**Instance 1 - Filling planning.md**

- *What I gave the AI:* The empty `planning.md` template and the chosen domain (AI/ML Research Onboarding), along with a request to suggest 10+ real sources and fill every section.
- *What it produced:* A fully populated planning.md including 12 real URLs, a chunking strategy with numeric justification, an embedding model choice with production tradeoff analysis, 5 testable evaluation questions with expected answers, an ASCII architecture diagram, and per-milestone AI tool plans.
- *What I changed or overrode:* The domain was my choice (selected from 5 options the AI suggested). The evaluation questions and expected answers were accepted as-is because they matched the specificity requirement from the rubric.

**Instance 2 - Building the full pipeline**

- *What I gave the AI:* The completed `planning.md` as specification context, plus a request to implement the five-stage pipeline (ingest → chunk → embed → retrieve → generate → Gradio interface).
- *What it produced:* Five files - `ingest.py`, `embed_store.py`, `generate.py`, `build_index.py`, and `app.py` - matching the architecture diagram and using the exact model names, chunk sizes, and top-k value from the spec. It also caught a relative-path bug in `build_index.py` (the `DOCUMENTS_FOLDER` path broke when the script was invoked from outside the project directory) and fixed it to use `Path(__file__).parent`.
- *What I changed or overrode:* The system prompt in `generate.py` was tightened to add an explicit anti-hallucination rule ("say so clearly rather than making things up") that the initial version expressed more vaguely. The source attribution footer in `app.py` was kept as generated - appending filenames to every response regardless of inline citation - because it provides a useful audit trail for retrieval quality during evaluation.
