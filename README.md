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

## Layout

```
download_data.py     OHR-Bench GT (law) + law QA pairs
chunkers.py          3 chunker adapters + char-span → page attribution
matching.py          vendored OHR-Bench normalize/lcs_score + doc+page gate
bench.py             resumable matrix runner (Drive-cached stages)
aggregate.py         CSV + paper tables + interaction-effect flags
chunk_embed_bench.ipynb  Colab driver
tests/run_tests.py   CPU-only test suite
```
