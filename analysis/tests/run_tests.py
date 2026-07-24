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
from analysis import aggregate_analysis, build_blind_eval, prepare_ground_truth, raters  # noqa: E402

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
    print("registry: cross-family OpenRouter candidates")
    check("2-4 models", 2 <= len(MODELS) <= 4, str(list(MODELS)))
    check("all openrouter backend", all(s["backend"] == "openrouter" for s in MODELS.values()))
    check("all candidates (no reference)", all(s["role"] == "candidate" for s in MODELS.values()))
    check("all families distinct",
          len({s["family"] for s in MODELS.values()}) == len(MODELS))
    check("Meta + Qwen retained",
          {s["family"] for s in MODELS.values()} >= {"Meta", "Qwen"})
    check("every model has a tier_note", all(s.get("tier_note") for s in MODELS.values()))
    check("ZDR provider prefs (DeepInfra-pinned, no fallback)",
          PROVIDER_PREFS == {"order": ["deepinfra"], "allow_fallbacks": False,
                             "data_collection": "deny", "zdr": True})


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
        check("2 docs x N models checkpoints",
              len(list((tmp / "analyses").glob("*.json"))) == 2 * len(ALL_MODELS))
        rec = json.loads((tmp / "analyses" / f"doc_a__{STRONG}.json").read_text())
        check("checkpoint metadata present",
              rec["ok"] and rec["prompt_version"] and rec["latency_label"])
        calls = []
        run_analyses(docs, ALL_MODELS, tmp, lambda s, m: (calls.append(s),
                                                          _fake_call_fn(s, m))[1])
        check("resume skips completed", calls == [])

        # a stale de-registered-model checkpoint must be ignored downstream
        (tmp / "analyses" / "doc_a__mixtral-8x22b.json").write_text(json.dumps(
            {"doc": "doc_a", "model": "mixtral-8x22b", "role": "candidate",
             "parsed": _good_obj(), "ok": True}))
        analyses = build_blind_eval.load_analyses(tmp)
        check("stale model filtered from blind load",
              all("mixtral-8x22b" not in per for per in analyses.values()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _StatusErr(Exception):
    def __init__(self, status):
        self.status_code = status
        self.message = f"HTTP {status} no endpoints match data policy"
        super().__init__(self.message)


class _RaisingClient:
    def __init__(self, exc):
        self._exc = exc
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(exc)))


def test_permanent_error_and_tolerance():
    print("404 fails fast (no retry); tolerate_failures records per-model + continues")
    call_fn = make_call_fn({"openrouter": _RaisingClient(_StatusErr(404))})
    try:
        call_fn("qwen-2.5-72b", [{"role": "user", "content": "x"}])
        check("404 raises", False)
    except RuntimeError as e:
        check("404 raised immediately with no retry", "404" in str(e) and "no retry" in str(e))

    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        # one model always 404s; the rest succeed. tolerate -> run completes.
        bad = [m for m in ALL_MODELS if m != STRONG][0]
        def fn(short, messages):
            if short == bad:
                raise RuntimeError(f"{MODELS[short]['id']}: HTTP 404 (no retry): data policy")
            return _fake_call_fn(short, messages)
        run_analyses({"d1": "t"}, ALL_MODELS, tmp, fn, tolerate_failures=True)
        recs = {p.stem.split("__")[1]: json.loads(p.read_text())
                for p in (tmp / "analyses").glob("*.json")}
        check("all models checkpointed under tolerance", len(recs) == len(ALL_MODELS))
        check("404 model recorded ok:false+call_failed",
              recs[bad]["ok"] is False and recs[bad]["call_failed"]
              and "404" in recs[bad]["error"])
        check("good models still ok", recs[STRONG]["ok"])
        # WITHOUT tolerance the same failure aborts
        shutil.rmtree(tmp / "analyses")
        raised = False
        try:
            run_analyses({"d1": "t"}, ALL_MODELS, tmp, fn, tolerate_failures=False)
        except RuntimeError:
            raised = True
        check("without tolerance the run aborts", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _rating_rows(pids, scores):
    """scores: dim -> value (constant) or dict pid->value."""
    rows = []
    for p in pids:
        r = {"presentation_id": p}
        for col, dim in raters.RATING_COLS.items():
            v = scores[dim]
            r[col] = v[p] if isinstance(v, dict) else v
        rows.append(r)
    return rows


def _write_rater_csv(cache, name, rows):
    with open(cache / f"blind_eval_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def test_multirater_load_and_agreement():
    print("multi-rater: load/validate, averaging, agreement stats")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        pids = [f"p{i}" for i in range(6)]
        # author_1: relevance varies; author_2 differs by 1 on relevance for p0..p2
        a1 = {"coherence": 5, "consistency": 4, "fluency": 5,
              "relevance": {p: (3 if i < 3 else 5) for i, p in enumerate(pids)}}
        a2 = {"coherence": 5, "consistency": 4, "fluency": 5,
              "relevance": {p: (4 if i < 3 else 5) for i, p in enumerate(pids)}}
        _write_rater_csv(tmp, "author_1", _rating_rows(pids, a1))
        _write_rater_csv(tmp, "author_2", _rating_rows(pids, a2))

        per_rater, averaged, source = raters.load_ratings(tmp)
        check("two raters detected", set(per_rater) == {"author_1", "author_2"}, source)
        check("averaged across raters (3 & 4 -> 3.5)", averaged["p0"]["relevance"] == 3.5)
        check("identical ratings average to themselves", averaged["p0"]["coherence"] == 5.0)

        stats = {r["dimension"]: r for r in raters.agreement_stats(per_rater)}
        check("agreement covers all 4 dimensions", len(stats) == 4)
        coh = stats["coherence"]
        check("ceiling dim: 100% exact", coh["exact_match_pct"] == 100.0)
        check("ceiling dim: r suppressed (zero variance)",
              coh["pearson_r"] is None and "zero variance" in coh["r_note"])
        rel = stats["relevance"]
        check("varying dim: exact 50%, within-1 100%",
              rel["exact_match_pct"] == 50.0 and rel["within1_pct"] == 100.0, str(rel))
        check("varying dim: per-rater means reported",
              rel["mean_author_1"] == 4.0 and rel["mean_author_2"] == 4.5, str(rel))
        check("varying dim: r computed", rel["pearson_r"] is not None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_multirater_validation():
    print("multi-rater: validation failures are loud")
    pids = [f"p{i}" for i in range(3)]
    good = {"coherence": 5, "consistency": 4, "fluency": 5, "relevance": 4}

    def expect_fail(label, mutate):
        tmp = Path(tempfile.mkdtemp(prefix="an_"))
        try:
            rows1 = _rating_rows(pids, good)
            rows2 = _rating_rows(pids, good)
            mutate(rows1, rows2)
            _write_rater_csv(tmp, "author_1", rows1)
            _write_rater_csv(tmp, "author_2", rows2)
            try:
                raters.load_ratings(tmp)
                check(label, False)
            except SystemExit as e:
                check(label, "RATING VALIDATION FAILED" in str(e))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    expect_fail("blank rating rejected",
                lambda r1, r2: r1[0].__setitem__("coherence_1to5", ""))
    expect_fail("out-of-range rating rejected",
                lambda r1, r2: r1[1].__setitem__("relevance_1to5", 7))
    expect_fail("mismatched presentation_id sets rejected",
                lambda r1, r2: r2[0].__setitem__("presentation_id", "ZZZ"))

    # unknown presentation_id vs the unblinding key
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        _write_rater_csv(tmp, "author_1", _rating_rows(pids, good))
        try:
            raters.load_ratings(tmp, valid_pids={"p0", "p1"})  # p2 unknown
            check("pid not in unblinding key rejected", False)
        except SystemExit as e:
            check("pid not in unblinding key rejected", "unblinding_key" in str(e))
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
        n_models = len(ALL_MODELS)
        check("blind rows = 2 docs x N models", n_blind == 2 * n_models, str(n_blind))

        blind = list(csv.DictReader(open(tmp / "blind_eval.csv")))
        key = {r["presentation_id"]: r for r in csv.DictReader(open(tmp / "unblinding_key.csv"))}
        expected_labels = {f"Output {c}" for c in "ABCDE"[:n_models]}
        check("labels span first N of A-E",
              {r["output_label"] for r in blind} == expected_labels)
        check("blind hides model", all("model" not in r for r in blind))
        check("SummEval columns",
              all(c in blind[0] for c in ("coherence_1to5", "consistency_1to5",
                                          "fluency_1to5", "relevance_1to5")))
        weak = next(m for m in ALL_MODELS if m != STRONG)
        model_of = {r["presentation_id"]: r["model"] for r in key.values()}
        for r in blind:
            base = 5 if model_of[r["presentation_id"]] == STRONG else 3
            for c in ("coherence_1to5", "consistency_1to5", "fluency_1to5", "relevance_1to5"):
                r[c] = base
        with open(tmp / "blind_eval.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=blind[0].keys()); w.writeheader(); w.writerows(blind)

        rows, wins, human_filled, gt_filled, n_bl, _ = aggregate_analysis.aggregate(tmp)
        check("all rated", human_filled == 2 * n_models)
        by = {r["model"]: r for r in rows}
        check("SummEval means un-blinded",
              by[STRONG]["coherence"] == 5.0 and by[weak]["coherence"] == 3.0)
        check("strong model wins risk F1",
              by[STRONG]["risk_f1"] == 1.0 and by[weak]["risk_f1"] < 1.0,
              str({m: by[m]["risk_f1"] for m in by}))
        check("entity F1 = 1.0 all", all(by[m]["entity_f1"] == 1.0 for m in by))
        check("all models in wins", set(wins) == set(ALL_MODELS))
        check("strong wins risk_f1 on both docs", wins[STRONG]["risk_f1"] == 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_registry()
    test_prompt_parsing()
    test_matching()
    test_context_check()
    test_openrouter_provider_prefs()
    test_multirater_load_and_agreement()
    test_multirater_validation()
    test_runner_resume_and_stale_filter()
    test_permanent_error_and_tolerance()
    test_prepare_ground_truth()
    test_blind_gate_and_aggregate()
    print(f"\nALL {PASS} CHECKS PASSED")
