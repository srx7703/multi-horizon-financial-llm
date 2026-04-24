"""
Phase 2 — 4-way BERTScore: base Gemma 2 27B / + v2 LoRA / base Gemma 4 31B / + v2 LoRA.

Reads preds_{base,v2}.json (Phase 1) and preds_gemma4_{base,v2g4}.json (Phase 2)
from ./preds/, computes BERTScore vs references, and writes a combined Phase 2
report with paired t-tests for both deltas.

Usage:
    python3 compute_bertscore_phase2.py

Output:
    evaluation_results_phase2.json   — full 4-way comparison for README
"""

import json
import os

import numpy as np
from bert_score import score as bert_score
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.path.join(BASE_DIR, "preds")
OUT = os.path.join(BASE_DIR, "evaluation_results_phase2.json")

MODES = {
    "base":        ("preds_base.json",         "Base Gemma 2 27B (no adapter)"),
    "v2":          ("preds_v2.json",           "Gemma 2 27B + v2 LoRA"),
    "base_gemma4": ("preds_gemma4_base.json",  "Base Gemma 4 31B (no adapter)"),
    "v2_gemma4":   ("preds_gemma4_v2g4.json",  "Gemma 4 31B + v2 LoRA"),
}

def load_preds(mode):
    fname, _ = MODES[mode]
    path = os.path.join(PREDS_DIR, fname)
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path} — generate on TPU and scp back.")
    return json.load(open(path))

def score_mode(mode):
    data = load_preds(mode)
    preds = [d["prediction"] for d in data]
    refs  = [d["reference"]  for d in data]
    P, R, F1 = bert_score(preds, refs, lang="en", verbose=False)
    _, label = MODES[mode]
    return {
        "label": label,
        "n": len(data),
        "bertscore_f1": round(F1.mean().item(), 4),
        "bertscore_p":  round(P.mean().item(), 4),
        "bertscore_r":  round(R.mean().item(), 4),
        "per_item_f1":  [round(f, 4) for f in F1.tolist()],
    }

def paired_test(base_f1s, adapter_f1s):
    diff = np.array(adapter_f1s) - np.array(base_f1s)
    t, p = stats.ttest_rel(adapter_f1s, base_f1s)
    ci_low, ci_high = stats.t.interval(
        0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff)
    )
    return {
        "mean_delta": round(float(diff.mean()), 4),
        "t_stat": round(float(t), 2),
        "p_value": round(float(p), 4),
        "ci_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()),
    }

def main():
    print(f"Computing 4-way BERTScore (RoBERTa-large)...\n")
    results = {mode: score_mode(mode) for mode in MODES}

    # Sanity: same N across all modes
    ns = {mode: r["n"] for mode, r in results.items()}
    assert len(set(ns.values())) == 1, f"Mismatched N per mode: {ns}"
    n = next(iter(ns.values()))

    # Paired tests
    tests = {
        "gemma2_v2_vs_base": paired_test(
            results["base"]["per_item_f1"],
            results["v2"]["per_item_f1"],
        ),
        "gemma4_v2_vs_base": paired_test(
            results["base_gemma4"]["per_item_f1"],
            results["v2_gemma4"]["per_item_f1"],
        ),
    }

    deltas_pct = {
        "gemma2_base_to_v2": round(
            (results["v2"]["bertscore_f1"] - results["base"]["bertscore_f1"])
            / results["base"]["bertscore_f1"] * 100, 2),
        "gemma4_base_to_v2": round(
            (results["v2_gemma4"]["bertscore_f1"] - results["base_gemma4"]["bertscore_f1"])
            / results["base_gemma4"]["bertscore_f1"] * 100, 2),
        "gemma2_v2_to_gemma4_v2": round(
            (results["v2_gemma4"]["bertscore_f1"] - results["v2"]["bertscore_f1"])
            / results["v2"]["bertscore_f1"] * 100, 2),
    }

    print("=" * 78)
    print(f"{'Model':<50} {'F1':>8} {'P':>8} {'R':>8}")
    print("-" * 78)
    for mode in MODES:
        r = results[mode]
        print(f"{r['label']:<50} {r['bertscore_f1']:>8.4f} "
              f"{r['bertscore_p']:>8.4f} {r['bertscore_r']:>8.4f}")
    print("=" * 78)
    print(f"\nDeltas (relative % of baseline):")
    for k, v in deltas_pct.items():
        print(f"  {k:<30} {v:+.2f}%")
    print(f"\nPaired t-tests (n={n}):")
    for k, t in tests.items():
        sig = "p<0.01" if t["p_value"] < 0.01 else f"p={t['p_value']}"
        print(f"  {k:<25} t={t['t_stat']:+.2f}, {sig}, "
              f"CI95=[{t['ci_95'][0]:+.4f},{t['ci_95'][1]:+.4f}], "
              f"wins={t['wins']}/{n}")

    summary = {
        "test_set_size": n,
        "results": results,
        "deltas_pct": deltas_pct,
        "paired_tests": tests,
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Report saved → {OUT}")

if __name__ == "__main__":
    main()
