#!/usr/bin/env python3
"""Build revision tables from existing v4 result JSON (no re-embed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT.parent / "BIMS-LEGAL-dataset" / "primary_results" / "bims_legal_v4"


def load(name: str) -> dict:
    return json.loads((V4 / name).read_text(encoding="utf-8"))


def failure_from_per_query(per_query: list[dict]) -> dict[str, int]:
    out = {"complete": 0, "incomplete": 0, "answer_only": 0, "session_miss": 0}
    for row in per_query:
        ah = bool(row.get("ah", False))
        sh = bool(row.get("sh", False))
        if ah and sh:
            out["complete"] += 1
        elif ah and not sh:
            out["incomplete"] += 1
        elif (not ah) and sh:
            out["answer_only"] += 1
        else:
            out["session_miss"] += 1
    return out


def main() -> None:
    disc = load("legalmem_disc_advice.json")
    para = load("legalmem_para_advice.json")
    disc_bm25 = load("legalmem_disc_bm25_joint.json")
    disc_joint = load("legalmem_disc_joint_qa.json")

    soft = disc["methods"]["soft_o2"]
    ft = failure_from_per_query(soft["per_query"])
    n = len(soft["per_query"])
    print("=== RQ1 failure taxonomy (LegalEp DISC, Soft O2, n=%d) ===" % n)
    for k, v in ft.items():
        print(f"  {k}: {v} ({100*v/n:.1f}%)")

    print("\n=== BM25-joint vs Soft O2 (LegalEp DISC advice) ===")
    for m in ["bm25_joint", "soft_o2"]:
        r = disc["methods"][m]
        print(
            f"  {m}: AH={r['ah']:.3f} SH={r['sh']:.3f} "
            f"U1={r['u1']:.3f} Uk={r['uk']:.3f} Ulast={r['ulast']:.3f}"
        )

    print("\n=== Joint QA vs Soft O2 (LegalEp DISC) ===")
    jq = disc_joint["methods"]["joint_qa"]
    print(
        f"  joint_qa: AH={jq['ah']:.3f} SH={jq['sh']:.3f} "
        f"U1={jq['u1']:.3f} Uk={jq['uk']:.3f}"
    )

    print("\n=== CAIL Uk comparison ===")
    cail = load("legalmem_cail_advice.json")
    for m in ["flatip", "soft_o2", "bm25_joint"]:
        if m in cail["methods"]:
            r = cail["methods"][m]
            print(f"  {m}: AH={r['ah']:.3f} Uk={r['uk']:.3f} nDCG={r.get('ndcg', 'n/a')}")

    out = {
        "failure_taxonomy_disc_soft_o2": ft,
        "bm25_joint_disc": {k: disc["methods"]["bm25_joint"][k] for k in ["ah", "sh", "u1", "uk", "ulast"]},
        "soft_o2_disc": {k: soft[k] for k in ["ah", "sh", "u1", "uk", "ulast"]},
    }
    out_path = ROOT / "paper" / "ipm" / "figures" / "revision_extracted.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
