"""
Local-side BERTScore computation. Reads the three prediction JSON files
produced by generate_on_tpu.py (after you scp them back), computes BERTScore
vs the reference answers, and writes a combined report.

Usage:
    # After scp'ing preds_base.json / preds_v1.json / preds_v2.json into ./preds/
    python3 compute_bertscore_v2.py

Output:
    evaluation_results_v2.json   — full report for README
"""

import json
import os

from bert_score import score as bert_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(BASE_DIR, "preds")
OUT = os.path.join(BASE_DIR, "evaluation_results_v2.json")

LABELS = {
    "base": "Base Gemma 2 27B (no adapter)",
    "v1":   "Gemma 2 27B + v1 LoRA (HF generic financial QA)",
    "v2":   "Gemma 2 27B + v2 LoRA (our SEC data)",
}
# Auto-discover available prediction files (skip v1 if we haven't produced it)
def _available_modes():
    out = []
    for m in ["base", "v1", "v2"]:
        if os.path.exists(os.path.join(PREDS_DIR, f"preds_{m}.json")):
            out.append(m)
    return out

def load_preds(mode):
    path = os.path.join(PREDS_DIR, f"preds_{mode}.json")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run generate_on_tpu.py --mode {mode} on TPU and scp back.")
    return json.load(open(path))

def score_mode(mode):
    data = load_preds(mode)
    preds = [d["prediction"] for d in data]
    refs  = [d["reference"] for d in data]
    P, R, F1 = bert_score(preds, refs, lang="en", verbose=False)
    return {
        "label": LABELS[mode],
        "n": len(data),
        "bertscore_f1": round(F1.mean().item(), 4),
        "bertscore_p":  round(P.mean().item(), 4),
        "bertscore_r":  round(R.mean().item(), 4),
        "per_item": [
            {
                "question": d["question"][:120],
                "reference": d["reference"][:200],
                "prediction": d["prediction"][:400],
                "f1": round(F1[i].item(), 4),
            }
            for i, d in enumerate(data)
        ],
    }

def main():
    modes = _available_modes()
    if "base" not in modes or "v2" not in modes:
        raise SystemExit(f"Need at least preds_base.json and preds_v2.json; found: {modes}")

    # Sanity check that available prediction files share the same question set
    sets = {mode: [d["question"] for d in load_preds(mode)] for mode in modes}
    first = sets[modes[0]]
    for m in modes[1:]:
        assert sets[m] == first, f"Prediction file {m} has different questions"

    print(f"Computing BERTScore (RoBERTa-large) for modes: {modes}\n")
    results = {mode: score_mode(mode) for mode in modes}

    base_f1 = results["base"]["bertscore_f1"]
    v2_f1   = results["v2"]["bertscore_f1"]

    print("=" * 70)
    print(f"{'Model':<55} {'F1':>8}")
    print("-" * 70)
    for mode in modes:
        r = results[mode]
        print(f"{r['label']:<55} {r['bertscore_f1']:>8.4f}")
    print("=" * 70)

    deltas = {"base_to_v2_pct": round((v2_f1 - base_f1) / base_f1 * 100, 2)}
    if "v1" in modes:
        v1_f1 = results["v1"]["bertscore_f1"]
        deltas["base_to_v1_pct"] = round((v1_f1 - base_f1) / base_f1 * 100, 2)
        deltas["v1_to_v2_pct"]   = round((v2_f1 - v1_f1)   / v1_f1   * 100, 2)

    summary = {
        "test_set_size": results["base"]["n"],
        "results": results,
        "deltas": deltas,
        "interpretation": (
            f"Base → v2: {deltas['base_to_v2_pct']:+.2f}% "
            f"(SEC-domain LoRA fine-tune vs base Gemma 2 27B)."
        ),
    }

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Report saved → {OUT}")
    print(f"\nDeltas:")
    for k, v in summary["deltas"].items():
        print(f"  {k}:  {v:+.2f}%")

if __name__ == "__main__":
    main()
