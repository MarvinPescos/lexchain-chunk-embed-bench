"""Three chunker adapters with comparable configs.

Target: ~512-token chunks, ~50-token overlap, all token counts measured with
the SAME tiktoken cl100k_base encoding so the size budget is comparable:

- langchain:  RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                  encoding_name="cl100k_base", chunk_size=512, chunk_overlap=50)
- chonkie:    RecursiveChunker (recursive recipe, default rules) with a
              cl100k_base token counter, chunk_size=512. Recursive chunking has
              no native overlap, so OverlapRefinery(context_size=50) is applied
              when available; the effective overlap is recorded in the config
              string either way.
- llamaindex: SentenceSplitter(chunk_size=512, chunk_overlap=50) with the
              cl100k_base tokenizer.

Each adapter returns chunks as dicts:
    {"text": str, "doc": stem, "start": char_off, "end": char_off, "pages": [..]}
Page attribution: documents are the "\\n".join of GT page texts; a chunk's pages
are every page whose char range intersects the chunk's [start, end) span.
"""

from __future__ import annotations

import time

import tiktoken

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def _locate(doc_text: str, pieces: list[str]) -> list[tuple[int, int]]:
    """Char spans for sequential (possibly overlapping) chunk texts.

    Splitters strip whitespace at chunk edges, so each chunk is a contiguous
    substring of the source; consecutive chunks start at increasing offsets.
    Falls back to a prefix search, then to abutting the previous chunk.
    """
    spans = []
    cursor = 0
    for piece in pieces:
        start = doc_text.find(piece, cursor)
        if start == -1:
            start = doc_text.find(piece)
        if start == -1:  # e.g. refinery-modified text: anchor by prefix
            prefix = piece[:60]
            start = doc_text.find(prefix, cursor)
            if start == -1:
                start = doc_text.find(prefix)
        if start == -1:
            start = spans[-1][1] if spans else 0
        spans.append((start, start + len(piece)))
        cursor = start + 1
    return spans


def chunk_langchain(doc_text: str) -> tuple[list[tuple[str, int, int]], str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    pieces = splitter.split_text(doc_text)
    spans = _locate(doc_text, pieces)
    config = (f"RecursiveCharacterTextSplitter.from_tiktoken_encoder("
              f"cl100k_base, chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})")
    return [(p, s, e) for p, (s, e) in zip(pieces, spans)], config


def chunk_chonkie(doc_text: str) -> tuple[list[tuple[str, int, int]], str]:
    from chonkie import RecursiveChunker

    chunker = RecursiveChunker(tokenizer="cl100k_base", chunk_size=CHUNK_SIZE)
    chunks = chunker.chunk(doc_text)
    overlap_note = "no overlap (OverlapRefinery unavailable)"
    try:
        from chonkie import OverlapRefinery

        # prefix context = preceding-chunk overlap, like the other two splitters;
        # start/end indices keep the core span, so page mapping is unaffected
        refinery = OverlapRefinery(
            tokenizer="cl100k_base", context_size=CHUNK_OVERLAP,
            method="prefix", merge=True,
        )
        chunks = refinery.refine(chunks)
        overlap_note = f"OverlapRefinery(context_size={CHUNK_OVERLAP}, method=prefix)"
    except Exception as e:  # keep the un-refined chunks, but say so
        overlap_note = f"no overlap (OverlapRefinery failed: {type(e).__name__})"
    out = []
    for c in chunks:
        start = getattr(c, "start_index", None)
        end = getattr(c, "end_index", None)
        if start is None or end is None:
            spans = _locate(doc_text, [c.text])
            start, end = spans[0]
        out.append((c.text, int(start), int(end)))
    config = (f"chonkie RecursiveChunker(tokenizer=cl100k_base, "
              f"chunk_size={CHUNK_SIZE}) + {overlap_note}")
    return out, config


def chunk_llamaindex(doc_text: str) -> tuple[list[tuple[str, int, int]], str]:
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=lambda t: _ENC.encode(t, disallowed_special=()),
    )
    pieces = splitter.split_text(doc_text)
    spans = _locate(doc_text, pieces)
    config = (f"SentenceSplitter(chunk_size={CHUNK_SIZE}, "
              f"chunk_overlap={CHUNK_OVERLAP}, tokenizer=cl100k_base)")
    return [(p, s, e) for p, (s, e) in zip(pieces, spans)], config


CHUNKERS = {
    "langchain": chunk_langchain,
    "chonkie": chunk_chonkie,
    "llamaindex": chunk_llamaindex,
}


def pages_for_span(page_offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    """Pages (by index into the doc's page list) intersecting [start, end)."""
    return [i for i, (ps, pe) in enumerate(page_offsets) if start < pe and end > ps]


def chunk_corpus(chunker_name: str, docs: dict[str, list[str]]):
    """docs: stem -> list of page texts. Returns (chunks, config_str, wall_s).

    Each chunk dict carries its GT page indices so OHR-Bench's doc+page
    provenance gate can be applied at eval time.
    """
    fn = CHUNKERS[chunker_name]
    chunks, config = [], None
    t0 = time.time()
    for stem in sorted(docs):
        pages = docs[stem]
        offsets, pos = [], 0
        for p in pages:
            offsets.append((pos, pos + len(p)))
            pos += len(p) + 1  # the "\n" joiner
        doc_text = "\n".join(pages)
        if not doc_text.strip():
            continue
        pieces, config = fn(doc_text)
        for text, start, end in pieces:
            if not text.strip():
                continue
            chunks.append({
                "text": text,
                "doc": stem,
                "start": start,
                "end": end,
                "pages": pages_for_span(offsets, start, end),
            })
    return chunks, config, time.time() - t0
