#!/usr/bin/env python3
"""Download the benchmark data (idempotent):

  data/gt/law/*.json   OHR-Bench ground-truth text, Law domain
                       (HF dataset opendatalab/OHR-Bench, retrieval.zip)
  data/qas_law.json    Law-domain QA pairs filtered from qas_v2.json
                       (OHR-Bench GitHub repo; 1,142 QAs over 95 docs)
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

QAS_URL = "https://raw.githubusercontent.com/opendatalab/OHR-Bench/main/data/qas_v2.json"

DATA_DIR = Path(__file__).parent / "data"


def download_gt():
    gt_dir = DATA_DIR / "gt" / "law"
    if gt_dir.exists() and any(gt_dir.glob("*.json")):
        print(f"gt already present: {len(list(gt_dir.glob('*.json')))} docs")
        return
    from huggingface_hub import hf_hub_download

    print("downloading retrieval.zip from HF opendatalab/OHR-Bench ...")
    zpath = hf_hub_download("opendatalab/OHR-Bench", "retrieval.zip", repo_type="dataset")
    gt_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        for member in zf.namelist():
            parts = Path(member).parts
            if member.endswith(".json") and "gt" in parts and "law" in parts:
                (gt_dir / Path(member).name).write_bytes(zf.read(member))
                n += 1
    print(f"extracted {n} gt files -> {gt_dir}")


def download_qas():
    out = DATA_DIR / "qas_law.json"
    if out.exists():
        print(f"qas already present: {len(json.loads(out.read_text()))} law QAs")
        return
    print(f"downloading qas_v2.json ...")
    with urllib.request.urlopen(QAS_URL, timeout=120) as r:
        qas = json.load(r)
    law = [q for q in qas if q["doc_name"].startswith("law/")]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(law, ensure_ascii=False, indent=1))
    tmp.replace(out)
    print(f"{len(law)} law QAs (of {len(qas)} total) -> {out}")


def main():
    download_gt()
    download_qas()
    gt = list((DATA_DIR / "gt" / "law").glob("*.json"))
    qas = json.loads((DATA_DIR / "qas_law.json").read_text())
    qa_docs = {q["doc_name"].split("/", 1)[1] for q in qas}
    gt_stems = {p.stem for p in gt}
    print(f"check: {len(gt)} gt docs, {len(qas)} QAs over {len(qa_docs)} docs, "
          f"{len(qa_docs & gt_stems)} matched")
    missing = qa_docs - gt_stems
    if missing:
        print(f"WARNING: {len(missing)} QA docs missing gt: {sorted(missing)[:5]}")


if __name__ == "__main__":
    main()
