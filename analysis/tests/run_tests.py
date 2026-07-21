#!/usr/bin/env python3
"""CPU-only tests for the 4+1 model analysis benchmark (fake LLM, no network).

Run:  python analysis/tests/run_tests.py
"""

import csv
import json
import shutil
import sys
import tempfile
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
import types  # noqa: E402

from analysis.data import EXCLUDED_MODELS, MODELS, REFERENCE_ROLE, context_check  # noqa: E402
from analysis.matching_entities import score_category  # noqa: E402
from analysis.analyze import assert_vram, make_call_fn, run_analyses  # noqa: E402
from analysis import aggregate_analysis, build_blind_eval, prepare_ground_truth  # noqa: E402

PASS = 0
ALL_MODELS = list(MODELS)  # 4 candidates + 1 reference
CANDIDATES = [m for m in ALL_MODELS if MODELS[m]["role"] == "candidate"]


def check(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  ok: {name}")


def _risk_flags(present_map=None):
    present_map = present_map or {}
    return [
        {"category": c, "present": c in present_map,
         "quote": present_map.get(c, "")}
        for c in RISK_CATEGORY_NAMES
    ]


def _good_obj(present_map=None):
    return {
        "summary": "The parties entered an agreement with obligations and fees.",
        "entities": {
            "parties": ["Acme Corp", "Beta LLC"],
            "dates": ["2020-06-02"],
            "monetary_amounts": ["$100"],
            "obligations": ["Acme must pay Beta"],
        },
        "risk_flags": _risk_flags(present_map),
    }


def test_prompt_parsing():
    print("prompt: checklist schema, parsing, think-stripping")
    msgs = build_messages("DOC")
    check("prompt contains all 12 categories",
          all(c in msgs[1]["content"] for c in RISK_CATEGORY_NAMES))

    good = json.dumps(_good_obj({"auto_renewal": "renews automatically each year"}))
    p = parse_analysis(good)
    check("parses clean json", p is not None)
    check("12 checklist entries in order",
          [f["category"] for f in p["risk_flags"]] == RISK_CATEGORY_NAMES)
    ar = next(f for f in p["risk_flags"] if f["category"] == "auto_renewal")
    check("present category keeps quote", ar["present"] and "renews" in ar["quote"])
    check("schema valid", validate_schema(p) == [])

    think = "<think>Let me reason...\nblah</think>\n" + good
    check("qwen think block stripped", parse_analysis(think) is not None)
    check("strip_thinking removes block", "<think>" not in strip_thinking(think))
    check("unterminated think dropped",
          strip_thinking("prefix <think> endless reasoning") == "prefix")

    fenced = "```json\n" + good + "\n```"
    check("parses fenced json", parse_analysis(fenced) is not None)
    check("unparseable -> None", parse_analysis("not json") is None)

    # widened parser: preamble + fenced JSON + trailing prose (small-model shape)
    messy = "Here is the analysis you requested:\n```json\n" + good + "\n```\nLet me know!"
    check("parses preamble+fence+trailing prose", parse_analysis(messy) is not None)
    # trailing commas (common small-model quirk) repaired
    trailing = '{"summary":"S","entities":{"parties":["A",],"dates":[],' \
               '"monetary_amounts":[],"obligations":[],},"risk_flags":[]}'
    p2 = parse_analysis(trailing)
    check("repairs trailing commas", p2 is not None and p2["entities"]["parties"] == ["A"])

    # EXACT llama3.1:8b bug: dropped "quote": key -> bare "" for absent categories
    # (both space and no-space variants in one payload)
    dropped = ('{"summary":"S","entities":{"parties":[],"dates":[],'
               '"monetary_amounts":[],"obligations":[]},"risk_flags":['
               '{"category":"exclusivity","present":false, ""},'
               '{"category":"non_compete","present":false,""}]}')
    p3 = parse_analysis(dropped)
    check("repairs dropped quote key (bare \"\")", p3 is not None, "still unparseable")
    excl = next(f for f in p3["risk_flags"] if f["category"] == "exclusivity")
    check("recovered absent category has quote key", excl["present"] is False
          and excl["quote"] == "")
    check("recovered payload is schema-valid", validate_schema(p3) == [])

    # normalization: missing/misordered categories get realigned; extras dropped
    partial = _good_obj()
    partial["risk_flags"] = [{"category": "governing_law", "present": True,
                              "quote": "laws of Virginia"}]
    n = normalize_analysis(partial)
    check("missing categories filled as absent",
          len(n["risk_flags"]) == 12 and
          sum(f["present"] for f in n["risk_flags"]) == 1)
    bad = _good_obj({"indemnification": ""})  # present without quote
    bad["risk_flags"][0]["quote"] = ""
    bad["risk_flags"][0]["present"] = True
    problems = validate_schema(normalize_analysis(bad))
    check("present-without-quote flagged", any("without a quote" in p for p in problems))


def test_matching():
    print("entity/risk matching + F1")
    s = score_category("parties", ["Acme Corp", "Beta LLC"], ["ACME CORP.", "Beta, LLC"])
    check("fuzzy party match tp=2", s["tp"] == 2 and s["f1"] == 1.0, str(s))
    s = score_category("parties", ["Acme Corp", "Ghost Inc"], ["Acme Corporation"])
    check("party precision/recall", s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 0, str(s))
    s = score_category("dates", ["June 2, 2020"], ["2020-06-02"])
    check("date format-invariant match", s["tp"] == 1, str(s))
    s = score_category("monetary_amounts", ["$4,162,000.00 grant"], ["4162000 dollars"])
    check("money numeric match", s["tp"] == 1, str(s))


def test_context_check():
    print("context safety (fail loudly, never truncate)")
    small = {"doc": "hello " * 100}
    report = context_check(small, ["llama3.1-8b"])
    check("small doc fits", report["per_model"]["llama3.1-8b"]["limit"] == 32768)
    huge = {"doc": "word " * 40000}  # ~40k tokens > every num_ctx
    try:
        context_check(huge, ["llama3.1-8b"])
        check("oversized doc rejected", False)
    except SystemExit as e:
        check("oversized doc rejected", "CONTEXT CHECK FAILED" in str(e))
    check("phi4 recorded as excluded with reason",
          "16k context" in EXCLUDED_MODELS["phi4-14b"]["reason"]
          and "phi4-14b" not in MODELS)
    check("candidate set is 4 families + reference (all T4-viable)",
          set(MODELS) == {"llama3.1-8b", "qwen3-14b", "mistral-nemo-12b",
                          "gemma3-12b", "llama-3.1-70b"})
    check("no candidate needs a VRAM gate (all T4-viable)",
          all("min_free_vram_gb" not in s for s in MODELS.values()))

    print("VRAM assertion (generic guard, tested via injected requirement)")
    MODELS["_vram_probe"] = {"backend": "ollama", "id": "probe", "role": "candidate",
                             "native_ctx": 4096, "num_ctx": 4096, "min_free_vram_gb": 17}
    try:
        assert_vram(["_vram_probe"], free_gb=22.0)  # enough
        check("sufficient VRAM passes", True)
        for free, label in ((14.5, "T4-sized rejected"), (None, "no GPU rejected")):
            try:
                assert_vram(["_vram_probe"], free_gb=free)
                check(label, False)
            except SystemExit as e:
                check(label, "VRAM CHECK FAILED" in str(e))
    finally:
        del MODELS["_vram_probe"]
    assert_vram(["llama3.1-8b"], free_gb=None)  # no requirement -> no check
    check("models without requirement skip vram check", True)


def test_prepare_ground_truth():
    print("prepare_ground_truth: blank template + readable dumps")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        docs = {"doc_alpha": "PARTY ONE and PARTY TWO agree. " * 20,
                "doc_beta": "Effective 2021-01-01. Fee is $500."}
        tmpl = prepare_ground_truth.write_ground_truth_template(tmp, docs)
        rows = list(csv.DictReader(open(tmpl)))
        check("template one blank row per doc", len(rows) == 2, str(len(rows)))
        check("template cols == aggregator's GT_LIST_COLS",
              prepare_ground_truth.GT_KEY_COLS == aggregate_analysis.GT_LIST_COLS)
        check("template gt cells are blank",
              all(r[c] == "" for r in rows for c in prepare_ground_truth.GT_KEY_COLS))
        dumps = prepare_ground_truth.write_doc_dumps(tmp, docs)
        check("one dump per doc + index", len(dumps) == 2
              and (tmp / "doc_texts" / "INDEX.txt").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# fake deterministic LLM: model-specific quality so aggregation is testable.
# llama3.1-8b gets everything right; others miss the auto_renewal risk.
def _fake_call_fn(short, messages):
    if short == "llama3.1-8b":
        obj = _good_obj({"auto_renewal": "renews automatically each year",
                         "governing_law": "laws of Virginia"})
    else:
        obj = _good_obj({"governing_law": "laws of Virginia"})
    raw = json.dumps(obj)
    if short == "qwen3-14b":  # simulate thinking-mode leakage
        raw = "<think>step by step...</think>" + raw
    return raw, {"completion_tokens": 42}, 0.5 if short != "llama-3.1-70b" else 2.0


class _StubClient:
    """Records the kwargs passed to chat.completions.create."""
    def __init__(self, content='{"x":1}'):
        self.captured = None
        self._content = content
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.captured = kwargs
        msg = types.SimpleNamespace(content=self._content)
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=usage)


def test_ollama_json_format():
    print("ollama backend requests constrained JSON; nim does not")
    ollama, nim = _StubClient(), _StubClient()
    call_fn = make_call_fn({"ollama": ollama, "nim": nim})
    call_fn("llama3.1-8b", [{"role": "user", "content": "hi"}])
    check("ollama gets response_format json_object",
          ollama.captured["response_format"] == {"type": "json_object"})
    check("ollama sets native format:json via extra_body",
          ollama.captured["extra_body"]["format"] == "json")
    check("ollama sets num_ctx",
          ollama.captured["extra_body"]["options"]["num_ctx"] == 32768)
    call_fn("llama-3.1-70b", [{"role": "user", "content": "hi"}])
    check("nim reference has no response_format constraint",
          "response_format" not in nim.captured)


def test_reference_tolerant():
    print("reference failure is caught (candidates still complete); retryable")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        docs = {"doc_a": "t"}

        def fn_ref_fails(short, messages):
            if MODELS[short]["role"] != "candidate":
                raise RuntimeError("403 Authorization failed")
            return _fake_call_fn(short, messages)

        run_analyses(docs, ALL_MODELS, tmp, fn_ref_fails)  # must NOT raise
        cand_ok = [json.loads((tmp / "analyses" / f"doc_a__{m}.json").read_text())["ok"]
                   for m in CANDIDATES]
        check("four candidates completed", all(cand_ok) and len(cand_ok) == 4)
        ref = json.loads((tmp / "analyses" / "doc_a__llama-3.1-70b.json").read_text())
        check("reference recorded ok:false + call_failed + error",
              ref["ok"] is False and ref["call_failed"] and "403" in ref["error"])

        # resume RE-ATTEMPTS the failed reference; candidates are skipped
        calls = []
        def fn_all_ok(short, messages):
            calls.append(short)
            return _fake_call_fn(short, messages)
        run_analyses(docs, ALL_MODELS, tmp, fn_all_ok)
        check("resume retries only the failed reference", calls == ["llama-3.1-70b"],
              str(calls))
        ref2 = json.loads((tmp / "analyses" / "doc_a__llama-3.1-70b.json").read_text())
        check("reference now succeeds on rerun", ref2["ok"] and not ref2["call_failed"])

        # a CANDIDATE hard failure must still fail loudly (abort)
        def fn_cand_fails(short, messages):
            if short == "gemma3-12b":
                raise RuntimeError("ollama server died")
            return _fake_call_fn(short, messages)
        raised = False
        try:
            run_analyses({"doc_b": "t"}, ALL_MODELS, tmp, fn_cand_fails)
        except RuntimeError:
            raised = True
        check("candidate failure aborts the run", raised)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_runner_resume():
    print("runner: checkpoint + resume + metadata")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        docs = {"doc_a": "text a", "doc_b": "text b"}
        run_analyses(docs, ALL_MODELS, tmp, _fake_call_fn)
        files = list((tmp / "analyses").glob("*.json"))
        check("10 checkpoints (2 docs x 5 models)", len(files) == 10, str(len(files)))
        rec = json.loads((tmp / "analyses" / "doc_a__qwen3-14b.json").read_text())
        check("think-leaked output parsed ok", rec["ok"], str(rec.get("schema_problems")))
        check("checkpoint has prompt_version+role+latency label",
              rec["prompt_version"] and rec["role"] == "candidate"
              and rec["latency_label"] == "colab_gpu")
        ref = json.loads((tmp / "analyses" / "doc_a__llama-3.1-70b.json").read_text())
        check("reference role recorded", ref["role"] == REFERENCE_ROLE)

        calls = []
        def counting_fn(short, msgs):
            calls.append(short)
            return _fake_call_fn(short, msgs)
        run_analyses(docs, ALL_MODELS, tmp, counting_fn)
        check("resume skips completed", calls == [], f"made {len(calls)} calls")
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
                        "risk_clauses": "auto renewal each year; governing law Virginia"})


def test_blind_gate_and_aggregate():
    print("GT-first guardrail + blind sheet (A-E) + aggregate + wins")
    tmp = Path(tempfile.mkdtemp(prefix="an_"))
    try:
        docs = {"doc_alpha": "t", "doc_beta": "t"}
        run_analyses(docs, ALL_MODELS, tmp, _fake_call_fn)

        # guardrail: refuses before the key exists
        try:
            build_blind_eval.build(tmp)
            check("guardrail blocks without GT key", False)
        except SystemExit as e:
            check("guardrail blocks without GT key", "GUARDRAIL" in str(e))
        _write_gt_key(tmp, docs)
        n_blind, _ = build_blind_eval.build(tmp)  # passes now
        check("blind rows = docs x 5 models", n_blind == 10, str(n_blind))

        blind = list(csv.DictReader(open(tmp / "blind_eval.csv")))
        key = {r["presentation_id"]: r for r in csv.DictReader(open(tmp / "unblinding_key.csv"))}
        check("labels span A-E",
              {r["output_label"] for r in blind} == {f"Output {c}" for c in "ABCDE"})
        check("blind rows hide model+role",
              all("model" not in r and "role" not in r for r in blind))
        check("SummEval columns present",
              all(c in blind[0] for c in ("coherence_1to5", "consistency_1to5",
                                          "fluency_1to5", "relevance_1to5",
                                          "hallucinations_notes")))
        check("key carries reference role",
              sum(r["role"] == REFERENCE_ROLE for r in key.values()) == 2)

        # fill ratings: give llama3.1-8b top scores, others lower
        model_of = {r["presentation_id"]: r["model"] for r in key.values()}
        for r in blind:
            base = 5 if model_of[r["presentation_id"]] == "llama3.1-8b" else 3
            for c in ("coherence_1to5", "consistency_1to5", "fluency_1to5", "relevance_1to5"):
                r[c] = base
        with open(tmp / "blind_eval.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=blind[0].keys())
            w.writeheader(); w.writerows(blind)

        rows, wins, human_filled, gt_filled, n_bl, _ = aggregate_analysis.aggregate(tmp)
        check("all rows rated", human_filled == 10, str(human_filled))
        by_model = {r["model"]: r for r in rows}
        check("SummEval means un-blinded",
              by_model["llama3.1-8b"]["coherence"] == 5.0
              and by_model["mistral-nemo-12b"]["coherence"] == 3.0)
        check("risk F1 rewards the complete model",
              by_model["llama3.1-8b"]["risk_f1"] == 1.0
              and by_model["mistral-nemo-12b"]["risk_f1"] < 1.0,
              str({m: by_model[m]["risk_f1"] for m in by_model}))
        check("entity F1 = 1.0 all (fake entities match key)",
              all(by_model[m]["entity_f1"] == 1.0 for m in by_model))
        check("latency recorded", by_model["llama-3.1-70b"]["mean_latency_s"] == 2.0)
        check("wins exclude reference", "llama-3.1-70b" not in wins)
        check("llama3.1-8b wins risk_f1 on both docs",
              wins["llama3.1-8b"]["risk_f1"] == 2, str(wins["llama3.1-8b"]))
        check("ties count for all (entity_f1)",
              all(wins[m]["entity_f1"] == 2 for m in CANDIDATES), str(wins))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_prompt_parsing()
    test_matching()
    test_context_check()
    test_ollama_json_format()
    test_reference_tolerant()
    test_prepare_ground_truth()
    test_runner_resume()
    test_blind_gate_and_aggregate()
    print(f"\nALL {PASS} CHECKS PASSED")
