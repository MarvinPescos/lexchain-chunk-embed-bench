# lexchain-chunk-embed-bench

Chunking × embedding retrieval benchmark for the LexChain capstone, on the **Law domain of [OHR-Bench](https://github.com/opendatalab/OHR-Bench)**: 95 ground-truth documents (page-level text) and 1,142 QA pairs (question + evidence context + evidence page). Companion to [lexchain-parser-bench](https://github.com/MarvinPescos/lexchain-parser-bench).

## Matrix (9 configs)

**Chunkers** — all targeting 512 tokens / 50 overlap, counted with the same `cl100k_base` tiktoken encoding (exact config strings are embedded in every result file and printed by `aggregate.py`):

| name | config |
|---|---|
| langchain | `RecursiveCharacterTextSplitter.from_tiktoken_encoder(cl100k_base, chunk_size=512, chunk_overlap=50)` |
| chonkie | `RecursiveChunker` (recursive recipe, cl100k counter, `chunk_size=512`) + `OverlapRefinery(context_size=50)` — recursive chunking has no native overlap; if the refinery is unavailable the config string says so |
| llamaindex | `SentenceSplitter(chunk_size=512, chunk_overlap=50, tokenizer=cl100k_base)` |

**Embedders** — all via sentence-transformers, normalized vectors, cosine retrieval:

| name | model | prefixes |
|---|---|---|
| e5-base-v2 | `intfloat/e5-base-v2` | `query: ` / `passage: ` (required by the model) |
| bge-base-en-v1.5 | `BAAI/bge-base-en-v1.5` | query: `Represent this sentence for searching relevant passages: `; passages bare |
| minilm-l6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | none |

## Evaluation

Evidence matching mirrors OHR-Bench's released retrieval eval (`src/tasks/retrieval.py` + `src/metric/common.py`, vendored in `matching.py` with attribution): a retrieved chunk counts toward a question only if it comes from the **correct document AND overlaps an evidence page**; text match is **word-level LCS ÷ evidence length** after their normalization.

- **Recall@1 / Recall@5 / MRR@5 / nDCG@10** — binary relevance: doc+page gate AND per-chunk LCS ≥ 0.7 (`--relevance-threshold`)
- **LCS@5** — OHR-Bench's exact unthresholded metric on the gated top-5 concatenation
- **coverage** — fraction of QAs with ≥1 relevant chunk anywhere (did the chunker preserve the evidence at all?)
- Ops: chunking time, embedding throughput (chunks/s), chunk count/size stats, index size (MB, float32)
- All 1,142 QAs count in every denominator (as in OHR-Bench), including the ~30 chart/formula ones whose evidence may be unmatchable in plain text.

`aggregate.py` emits `matrix.csv` + three markdown tables (full 9-row matrix, chunker marginals averaged over embedders, embedder marginals averaged over chunkers) and flags **interaction effects**: any cell whose Recall@5 deviates from its additive marginals (row mean + column mean − grand mean) by more than 0.02.

## Run it (Colab free T4)

Open `chunk_embed_bench.ipynb` in Colab → T4 runtime → run cells: mount Drive + install (pinned deps; Colab's preinstalled torch is untouched) → download data → **5-doc smoke test + full-run projection** → full run → aggregate. Every stage (chunks / query embeddings / chunk embeddings / per-config results) is cached to `MyDrive/lexchain_bench/` atomically, so disconnects lose nothing and reruns resume. Smoke-test caches carry an `_n5` suffix and never collide with the full run.

## Local development (no GPU)

```bash
pip install -r requirements-dev.txt   # chunking libs + tiktoken, NO torch
python tests/run_tests.py             # chunkers, matching, metrics, resume logic
```

Tests use a deterministic bag-of-words `fake` embedder, so the whole pipeline runs on CPU in seconds.

## RAG generation benchmark (justify the LLM choice)

A second, downstream experiment: freeze the retrieval pipeline to this benchmark's winners (**LangChain splitter + e5-base-v2**, reusing the cached embeddings) and vary **only the generation LLM**. For each of the 1,142 Law QAs: retrieve top-k chunks, prompt the model with OHR-Bench's exact QA prompt, and score the answer with **OHR-Bench's own generation metric** (`ohr_gen_eval.py`, vendored):

- **OHR-Bench F1** (headline) — token-overlap F1 after normalization (lowercase, strip punctuation/articles). Yes/No/"noanswer" answers must match exactly or score 0.
- **Accuracy (EM)** — strict normalized equality; `acc_contains` (gold ⊆ prediction) reported as a forgiving supplement.
- Answer-format notes: brief answers wrapped in `<response>…</response>`, `"Not answerable"` abstention, punctuation stripping means numeric answers match on digit-strings. Metrics count all questions incl. table/formula/chart evidence (a real RAG miss counts against the model).

**Provider: Groq** (`GROQ_API_KEY`). Models are the durable post-Llama set — `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.6-27b` — because Groq's Llama 3.1/3.3 are EOL Aug 16 2026 (Llama 3.1 70B already removed). The runner **checkpoints after every question** and handles rate limits: a per-minute 429 backs off and retries; a daily-limit 429 stops cleanly so the same command resumes the next day. Free-tier 70B-class caps (~100–200K tokens/day) make a **stratified ~200 sample** the practical default — the smoke cell prints paid cost and free-tier calendar-day projections for both full and sample before you commit.

Run `rag_gen_bench.ipynb` on Colab: smoke (8×3) + projection → approve → `rag_generate.py --sample 200` (resumable) → `gen_aggregate.py`. Outputs to Drive: `gen_results_table.md` (model | F1 | accuracy | latency | cost/1k, winner marked as the system result), `gen_matrix.csv`, and `human_review.csv` (per-answer auto scores + blank `human_correct`, ~40 rows flagged `spot_check`). After the team fills it, `gen_aggregate.py --with-human <csv>` reports auto-vs-human agreement + Cohen's κ.

```bash
python tests/run_gen_tests.py   # scoring, sampling, retrieval, resume, backoff, aggregate — no API/torch
```

## Layout

```
download_data.py     OHR-Bench GT (law) + law QA pairs
chunkers.py          3 chunker adapters + char-span → page attribution
matching.py          vendored OHR-Bench normalize/lcs_score + doc+page gate
bench.py             resumable chunk×embed matrix runner (Drive-cached stages)
aggregate.py         CSV + paper tables + interaction-effect flags
chunk_embed_bench.ipynb  Colab driver (chunk×embed)
ohr_gen_eval.py      vendored OHR-Bench generation scoring (F1/EM) + QA prompt
groq_client.py       Groq client: rate-limit-aware backoff, daily-limit handling
rag_generate.py      resumable generation runner (fixed pipeline, vary LLM)
gen_estimate.py      full-run time/cost projection from the smoke test
gen_aggregate.py     model table + winner + human-review CSV + agreement
rag_gen_bench.ipynb  Colab driver (generation)
tests/               CPU-only test suites (run_tests.py, run_gen_tests.py)
```
