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

## LLM document-analysis comparison (`analysis/`)

A human-judged experiment answering **which model is best at legal analysis** — the single variable is the model. Three ~70B-tier instruct models from different families, all accessed through the **same OpenRouter API** under a **zero-data-retention** policy (API-only, no self-hosting, which removes the size-bias confound): `meta-llama/llama-3.3-70b-instruct` (Meta, 70B dense), `qwen/qwen-2.5-72b-instruct` (Qwen, 72B dense), `mistralai/mixtral-8x22b-instruct` (Mistral, MoE ~39B active — ~70B-class compute; tier caveat noted in `models_meta.json`). Every request sends `provider = {data_collection:"deny", allow_fallbacks:false, zdr:true}` — zero retention, and fails loudly rather than routing to a non-compliant provider. (Disable prompt logging in your OpenRouter account too.) One frozen prompt (v2.0-cuad-checklist: summary + entities + a fixed 12-category risk checklist derived from CUAD's expert clause categories, present/absent + verbatim quote), temperature 0, constrained JSON decoding (`response_format=json_object`), identical schema for all three, over the deterministic 10-doc Law sample (seed 42) → 30 checkpointed, resumable outputs with per-call API latency.

Safeguards: a **context-safety check** tokenizes every document + prompt and refuses to run if anything would truncate (Qwen's 32k window is the tightest and still clears the 21.6k worst case); schema validation with retry + raw-failure logging; a **ground-truth-first guardrail** — the blind sheet cannot be generated until the hand-authored `ground_truth_key.csv` exists, so the key is always written blind to model outputs; and **stale-checkpoint filtering** — outputs from de-registered models are ignored so a previous model set can never leak into results. Human ratings use the **SummEval** dimensions (coherence/consistency/fluency/relevance, 1–5) on a 30-row blind sheet (Output A–C labels, identities only in `unblinding_key.csv`); entity/risk F1 is scored against the hand-authored key. Final table sorts by risk F1, plus per-document win counts (n=10 — wins + means, no significance tests).

The deliverable is a **blind human-scoring scaffold**, not just auto-metrics:

```
analysis/analyze.py            resumable runner (checkpoint per doc×model, backoff, /v1/models validation)
analysis/prompt.py             the single fixed prompt + JSON schema + robust parser
analysis/build_blind_eval.py   blind_eval.csv (30 rows, identities hidden as Output A/B/C, seed 42),
                               unblinding_key.csv, ground_truth_key_template.csv
analysis/matching_entities.py  fuzzy entity/risk matching (org-suffix aware) for P/R/F1
analysis/aggregate_analysis.py un-blind → results.md: model | coherence | consistency | fluency | relevance | entity F1 | risk F1 | latency
analysis_compare.ipynb         Colab driver (secret OPENROUTER_API_KEY; 6-call smoke → 30-call full run)
analysis/tests/run_tests.py    CPU-only, fake deterministic LLM (no key, no network)
```

Run `analysis_compare.ipynb` on Colab (add `OPENROUTER_API_KEY` to Colab Secrets — no GPU needed). Rate the blind sheet and fill `ground_truth_key.csv` before running the aggregate cell. Local test: `python analysis/tests/run_tests.py`.
