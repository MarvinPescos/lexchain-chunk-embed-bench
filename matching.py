"""Evidence matching, mirroring OHR-Bench's retrieval evaluation exactly.

`normalize_answer` and `lcs_score` are vendored from OHR-Bench
(src/metric/common.py, https://github.com/opendatalab/OHR-Bench, Apache-2.0):
word-level Longest Common Subsequence between the normalized evidence (A, gold)
and the normalized retrieved text (B), returned as |LCS| / |A| in [0, 1] —
i.e. the fraction of evidence words recoverable, in order, from the text.

OHR-Bench applies this after gating retrieved chunks on the correct document
AND evidence page(s) (src/tasks/retrieval.py); `evidence_pages` /
`gate_chunks` reproduce that gate. Our binary hit for Recall/MRR/nDCG adds a
threshold on the per-chunk LCS (OHR-Bench itself reports the unthresholded
average, which bench.py also reports as LCS@5).
"""

from __future__ import annotations

import re
import string


def normalize_answer(s: str) -> str:
    """Vendored from OHR-Bench: lowercase, drop articles/punctuation, fix whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def lcs_score(prediction: str, ground_truth: str) -> float:
    """Vendored from OHR-Bench: word-level LCS(gold, pred) / len(gold words)."""
    A = normalize_answer(ground_truth).split()
    B = normalize_answer(prediction).split()
    if len(A) == 0:
        return 0.5  # OHR-Bench's convention for empty gold evidence
    prev = [0] * (len(B) + 1)
    for i in range(1, len(A) + 1):
        cur = [0] * (len(B) + 1)
        a = A[i - 1]
        for j in range(1, len(B) + 1):
            if a == B[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(B)] / len(A)


def evidence_text(qa: dict) -> str:
    """evidence_context is a string or a list of strings (joined like OHR-Bench)."""
    ev = qa["evidence_context"]
    return "\n".join(ev) if isinstance(ev, list) else ev


def evidence_pages(qa: dict) -> set[int]:
    pages = qa["evidence_page_no"]
    if not isinstance(pages, list):
        pages = [pages]
    return {int(p) for p in pages}


def qa_doc_stem(qa: dict) -> str:
    """'law/DUDE_39df...' -> 'DUDE_39df...'."""
    return qa["doc_name"].split("/", 1)[1]


def gate_chunks(chunks: list[dict], qa: dict, chunk_ids: list[int]) -> list[int]:
    """OHR-Bench's provenance gate: chunk must come from the QA's doc and
    overlap at least one evidence page."""
    doc = qa_doc_stem(qa)
    pages = evidence_pages(qa)
    return [
        cid for cid in chunk_ids
        if chunks[cid]["doc"] == doc and pages.intersection(chunks[cid]["pages"])
    ]
