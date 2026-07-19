# LexChain summarization comparison

Compares three summarization approaches for LexChain's document-analysis component on OHR-Bench Law documents, with a two-layer evaluation. Part of [lexchain-chunk-embed-bench](../README.md); reuses that repo's OHR-Bench Law data (`../data/gt/law`, via `../download_data.py`).

## Approaches (the only variable)

| approach | type | implementation |
|---|---|---|
| `textrank` | extractive | `sumy` TextRankSummarizer (PageRank over a sentence-similarity graph) |
| `lexrank` | extractive | `sumy` LexRankSummarizer (eigenvector centrality on a TF-IDF cosine graph) |
| `llm` | abstractive (**our system**) | Llama 3.1 70B via **NVIDIA NIM** (`meta/llama-3.1-70b-instruct`), fixed prompt, temp 0.2 |

**Why LexRank as the 2nd baseline:** it shares `sumy`'s interface with TextRank, so the two differ only in algorithm (same sentence-splitting/tokenization) — a clean apples-to-apples extractive contrast.

**Provider:** NVIDIA NIM is LexChain's actual deployment (free NVIDIA Developer key, ~40 RPM — ample for 13 docs). Set `NVIDIA_API_KEY`. Fallback: `LLM_PROVIDER=groq` + `GROQ_API_KEY` (Llama 3.3 70B) if NIM is impractical. Extractive target = 8 sentences; LLM prompt targets ~180 words (recorded per summary).

## Data

Deterministic sample of **13 docs** (`random.Random(42)` over the 95 sorted Law stems); the chosen stems are saved to `sample_docs.txt`.

## Evaluation — two layers

**Layer 1 — automated PROXY metrics** (`evaluate_summ.py`), computed against the source doc (reference-free): ROUGE-1/2/L, BERTScore, summary length, compression ratio. **These are proxies, not a quality verdict** — extractive methods inflate ROUGE by copying source sentences verbatim, they run longer than the LLM at `sentence_count=8` (raising ROUGE further), and BERTScore truncates long docs to 512 tokens.

**Layer 2 — blind human evaluation** (`build_blind_eval.py`): `blind_eval.csv`, one row per (doc, approach) with the **approach hidden** — per doc the 3 summaries are randomly permuted to neutral "Summary 1/2/3" labels (no learnable global pattern), `doc_reference` visible so raters open the source to judge, blank 1–5 columns for coverage / factual_accuracy / fluency + notes. `unblinding_key.csv` maps rows back to approaches — **keep it separate until ratings are done**. `aggregate_human.py` joins the filled ratings with the key and the auto metrics into the final table.

**Final table** (`summ_final_results.{csv,md}`): `approach | ROUGE-L | BERTScore | avg human coverage | avg human accuracy | avg human fluency | mean length | compression`. **Human scores are the primary basis for the conclusion; ROUGE is a flagged proxy.**

## Run it (Colab)

Open `summ_compare.ipynb` (T4 runtime for BERTScore; `NVIDIA_API_KEY` in the Secrets panel). Cells: install + data → 2-doc smoke test + projected time → full run (resumable, Drive-checkpointed) → auto metrics → blind CSV → final table. Rerun `aggregate_human.py` after raters fill `blind_eval.csv`.

## Local development (no GPU / no API key)

```bash
python tests/test_summ.py   # extractive + ROUGE + blind scaffold + resume (fake LLM)
```

Every summary is checkpointed to the cache dir (`MyDrive/lexchain_bench/summarization/` on Colab, `.cache_summ/` locally); reruns skip completed work.
