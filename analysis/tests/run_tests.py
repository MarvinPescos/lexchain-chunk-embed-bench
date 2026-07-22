#!/usr/bin/env python3
"""CPU-only tests for the 3-model OpenRouter (ZDR) analysis benchmark.

Fake LLM, no network, no key. Run:  python analysis/tests/run_tests.py
"""

import csv
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from analysis.prompt import (  # noqa: E402
    RISK_CATEGORY_NAMES,
    build_messages,
    normalize_analysis,
    parse_analysis,
    strip_thinking,
    validate_schema,
)
from analysis.data import MODELS, PROVIDER_PREFS, context_check  # noqa: E402
from analysis.matching_entities import score_category  # noqa: E402
from analysis.analyze import assert_vram, make_call_fn, run_analyses  # noqa: E402
from analysis import aggregate_analysis, build_blind_eval, prepare_ground_truth  # noqa: E402

PASS = 0
ALL_MODELS = list(MODELS)  # 3 candidates, all OpenRouter
STRONG = "llama-3.3-70b"  # the model our fake LLM makes "best"


def check(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  ok: {name}")


def _risk_flags(present_map=None):
    present_map = present_map or {}
    return [{"category": c, "present": c in present_map, "quote": present_map.get(c, "")}
            for c in RISK_CATEGORY_NAMES]


def _good_obj(present_map=None):
    return {
        "summary": "The parties entered an agreement with obligations and fees.",
        "entities": {"parties": ["Acme Corp", "Beta LLC"], "dates": ["2020-06-02"],
                     "monetary_amounts": ["$100"], "obligations": ["Acme must pay Beta"]},
        "risk_flags": _risk_flags(present_map),
    }


def test_registry():
    print("registry: 3 cross-family OpenRouter candidates")
    check("exactly 3 models", len(MODELS) == 3, str(list(MODELS)))
    check("all openrouter backend", all(s["backend"] == "openrouter" for s in MODELS.values()))
    check("all candidates (no reference)", all(s["role"] == "candidate" for s in MODELS.values()))
    check("three distinct families",
          {s["family"] for s in MODELS.values()} == {"Meta", "Qwen", "Mistral"})
    check("ZDR provider prefs", PROVIDER_PREFS == {"data_collection": "deny",
          "allow_fallbacks": False, "zdr": True})


def test_prompt_parsing():
    print("prompt: checklist schema, parsing, repairs")
    check("prompt has all 12 categories",
          all(c in build_messages("DOC")[1]["content"] for c in RISK_CATEGORY_NAMES))
    good = json.dumps(_good_obj({"auto_renewal": "renews each year"}))
    p = parse_analysis(good)
    check("parses clean json", p is not None)
    check("12 checklist entries in order",
          [f["category"] for f in p["risk_flags"]] == RISK_CATEGORY_NAMES)
    check("schema valid", validate_schema(p) == [])
    check("think block stripped", parse_analysis("<think>x</think>" + good) is not None)
    check("strip_thinking works", "<think>" not in strip_thinking("<think>a</think>b"))
    check("fenced json", parse_analysis("```json\n" + good + "\n```") is not None)
    check("preamble+fence+trailing",
          parse_analysis("Here:\n```json\n" + good + "\n```\nDone") is not None)
    # llama's dropped-quote-key bug: {"present":false, ""}
    dropped = ('{"summary":"S","entities":{"parties":[],"dates":[],"monetary_amounts":[],'
               '"obligations":[]},"risk_flags":[{"category":"exclusivity","present":false, ""},'
               '{"category":"non_compete","present":false,""}]}')
    p3 = parse_analysis(dropped)
    check("repairs dropped quote key", p3 is not None and validate_schema(p3) == [])
    check("unparseable -> None", parse_analysis("not json") is None)
    n = normalize_analysis({"summary": "x", "risk_flags": [
        {"category": "governing_law", "present": True, "quote": "VA law"}]})
    check("missing categories filled absent",
          len(n["risk_flags"]) == 12 and sum(f["present"] for f in n["risk_flags"]) == 1)


def test_matching():
    print("entity/risk matching + F1")
    s = score_category("parties", ["Acme Corp", "Beta LLC"], ["ACME CORP.", "Beta, LLC"])
    check("fuzzy party match tp=2", s["tp"] == 2 and s["f1"] == 1.0, str(s))
    check("date format-invariant",
          score_category("dates", ["June 2, 2020"], ["2020-06-02"])["tp"] == 1)
    check("money numeric",
          score_category("monetary_amounts", ["$4,162,000.00 grant"], ["4162000"])["tp"] == 1)


def test_context_check():
    print("context safety (qwen's 32k fits; oversized rejected)")
    docs = {"d": "hello " * 100}
    rep = context_check(docs, ["qwen-2.5-72b"])
    check("qwen limit is 32768", rep["per_model"]["qwen-2.5-72b"]["limit"] == 32768)
    try:
        context_check({"d": "word " * 40000}, ["qwen-2.5-72b"])
        check("oversized rejected", False)
    except SystemExit as e:
        check("oversized rejected", "CONTEXT CHECK FAILED" in str(e))
    # assert_vram is a no-op now (no model needs it); still works via injection
    check("no model needs a VRAM gate",
          all("min_free_vram_gb" not in s for s in MODELS.values()))
    MODELS["_probe"] = {"backend": "openrouter", "id": "p", "role": "candidate",
                        "native_ctx": 4096, "num_ctx": None, "min_free_vram_gb": 17}
    try:
        try:
            assert_vram(["_probe"], free_gb=10.0)
            check("vram guard still functions", False)
        except SystemExit:
            check("vram guard still functions", True)
    finally:
        del MODELS["_probe"]


class _StubClient:
    def __init__(self, content='{"x":1}'):
        self.captured = None
        self._content = content
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.captured = kwargs
        msg = types.SimpleNamespace(content=self._content)
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


def test_openrouter_provider_prefs():
    print("openrouter: ZDR provider prefs + constrained JSON on every call")
    stub = _StubClient()
    call_fn = make_call_fn({"openrouter": stub})
    call_fn("qwen-2.5-72b", [{"role": "user", "content": "hi"}])
    check("response_format json_object",
          stub.captured["response_format"] == {"type": "json_object"})
    check("provider prefs = ZDR/deny/no-fallback",
          stub.captured["extra_body"]["provider"] == PROVIDER_PREFS)
    check("temperature 0", stub.captured["temperature"] == 0)


def _fake_call_fn(short, messages):
    # STRONG model flags auto_renewal (matches the key); others miss it
    present = {"governing_law": "VA law"}
    if short == STRONG:
        present["auto_renewal"] = "renews each year"
    return json.dumps(_good_obj(present)), {"completion_tokens": 42}, \
        (1.0 if short == STRONG else 2.0)


def test_runner_resume_and_stale_filter():
    print("runner: checkpoints, resume, stale-model filtering")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        docs = {"doc_a": "t", "doc_b": "t"}
        run_analyses(docs, ALL_MODELS, tmp, _fake_call_fn)
        check("6 checkpoints (2 docs x 3 models)",
              len(list((tmp / "analyses").glob("*.json"))) == 6)
        rec = json.loads((tmp / "analyses" / f"doc_a__{STRONG}.json").read_text())
        check("checkpoint metadata present",
              rec["ok"] and rec["prompt_version"] and rec["latency_label"] == "colab_gpu"
              or rec["latency_label"])  # label set by caller default
        calls = []
        run_analyses(docs, ALL_MODELS, tmp, lambda s, m: (calls.append(s),
                                                          _fake_call_fn(s, m))[1])
        check("resume skips completed", calls == [])

        # a stale de-registered-model checkpoint must be ignored downstream
        (tmp / "analyses" / "doc_a__gemma3-12b.json").write_text(json.dumps(
            {"doc": "doc_a", "model": "gemma3-12b", "role": "candidate",
             "parsed": _good_obj(), "ok": True}))
        analyses = build_blind_eval.load_analyses(tmp)
        check("stale model filtered from blind load",
              all("gemma3-12b" not in per for per in analyses.values()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_candidate_failure_aborts():
    print("a hard call failure fails loudly (no silent leak)")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        def fn(short, messages):
            if short == "mixtral-8x22b":
                raise RuntimeError("no ZDR provider available")
            return _fake_call_fn(short, messages)
        raised = False
        try:
            run_analyses({"doc_a": "t"}, ALL_MODELS, tmp, fn)
        except RuntimeError:
            raised = True
        check("run aborts on hard failure", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_ground_truth():
    print("prepare_ground_truth: blank template + dumps")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        docs = {"doc_alpha": "PARTY ONE and PARTY TWO. " * 20, "doc_beta": "Fee $500."}
        tmpl = prepare_ground_truth.write_ground_truth_template(tmp, docs)
        rows = list(csv.DictReader(open(tmpl)))
        check("one blank row per doc, cols match aggregator",
              len(rows) == 2 and prepare_ground_truth.GT_KEY_COLS ==
              aggregate_analysis.GT_LIST_COLS)
        dumps = prepare_ground_truth.write_doc_dumps(tmp, docs)
        check("dumps + index", len(dumps) == 2 and (tmp / "doc_texts" / "INDEX.txt").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_gt_key(tmp, docs):
    with open(tmp / "ground_truth_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc_reference"] + aggregate_analysis.GT_LIST_COLS)
        w.writeheader()
        for d in docs:
            w.writerow({"doc_reference": d, "parties": "Acme Corp; Beta LLC",
                        "dates": "2020-06-02", "monetary_amounts": "$100",
                        "obligations": "Acme must pay Beta",
                        "risk_clauses": "auto renewal each year; governing law VA"})


def test_blind_gate_and_aggregate():
    print("GT-first guardrail + blind (A-C) + SummEval + aggregate + wins")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        docs = {"doc_alpha": "t", "doc_beta": "t"}
        run_analyses(docs, ALL_MODELS, tmp, _fake_call_fn)
        try:
            build_blind_eval.build(tmp)
            check("guardrail blocks without GT key", False)
        except SystemExit as e:
            check("guardrail blocks without GT key", "GUARDRAIL" in str(e))
        _write_gt_key(tmp, docs)
        n_blind, _ = build_blind_eval.build(tmp)
        check("blind rows = 2 docs x 3 models", n_blind == 6, str(n_blind))

        blind = list(csv.DictReader(open(tmp / "blind_eval.csv")))
        key = {r["presentation_id"]: r for r in csv.DictReader(open(tmp / "unblinding_key.csv"))}
        check("labels span A-C only",
              {r["output_label"] for r in blind} == {"Output A", "Output B", "Output C"})
        check("blind hides model", all("model" not in r for r in blind))
        check("SummEval columns",
              all(c in blind[0] for c in ("coherence_1to5", "consistency_1to5",
                                          "fluency_1to5", "relevance_1to5")))
        model_of = {r["presentation_id"]: r["model"] for r in key.values()}
        for r in blind:
            base = 5 if model_of[r["presentation_id"]] == STRONG else 3
            for c in ("coherence_1to5", "consistency_1to5", "fluency_1to5", "relevance_1to5"):
                r[c] = base
        with open(tmp / "blind_eval.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=blind[0].keys()); w.writeheader(); w.writerows(blind)

        rows, wins, human_filled, gt_filled, n_bl, _ = aggregate_analysis.aggregate(tmp)
        check("all rated", human_filled == 6)
        by = {r["model"]: r for r in rows}
        check("SummEval means un-blinded",
              by[STRONG]["coherence"] == 5.0 and by["qwen-2.5-72b"]["coherence"] == 3.0)
        check("strong model wins risk F1",
              by[STRONG]["risk_f1"] == 1.0 and by["mixtral-8x22b"]["risk_f1"] < 1.0,
              str({m: by[m]["risk_f1"] for m in by}))
        check("entity F1 = 1.0 all", all(by[m]["entity_f1"] == 1.0 for m in by))
        check("all 3 in wins (no reference excluded)", set(wins) == set(ALL_MODELS))
        check("strong wins risk_f1 on both docs", wins[STRONG]["risk_f1"] == 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_registry()
    test_prompt_parsing()
    test_matching()
    test_context_check()
    test_openrouter_provider_prefs()
    test_runner_resume_and_stale_filter()
    test_candidate_failure_aborts()
    test_prepare_ground_truth()
    test_blind_gate_and_aggregate()
    print(f"\nALL {PASS} CHECKS PASSED")
