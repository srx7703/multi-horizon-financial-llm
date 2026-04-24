"""Schema tests for the committed BERTScore report."""
import json
import os


def test_eval_report_exists_and_valid(repo_root):
    path = os.path.join(repo_root, "evaluation_results_v2.json")
    assert os.path.exists(path), "evaluation_results_v2.json missing"
    report = json.load(open(path))
    assert "results" in report
    assert "deltas" in report
    assert "test_set_size" in report


def test_eval_report_has_base_and_v2(repo_root):
    report = json.load(open(os.path.join(repo_root, "evaluation_results_v2.json")))
    for mode in ("base", "v2"):
        assert mode in report["results"]
        m = report["results"][mode]
        for key in ("label", "n", "bertscore_f1", "bertscore_p", "bertscore_r"):
            assert key in m, f"{mode} missing {key}"
        assert 0 < m["bertscore_f1"] < 1


def test_v2_beats_base(repo_root):
    report = json.load(open(os.path.join(repo_root, "evaluation_results_v2.json")))
    assert (
        report["results"]["v2"]["bertscore_f1"] > report["results"]["base"]["bertscore_f1"]
    ), "LoRA adapter should improve over base model"


def test_predictions_files_match_eval_size(repo_root):
    report = json.load(open(os.path.join(repo_root, "evaluation_results_v2.json")))
    n = report["test_set_size"]
    for mode in ("base", "v2"):
        path = os.path.join(repo_root, "preds", f"preds_{mode}.json")
        data = json.load(open(path))
        assert len(data) == n, f"preds_{mode}.json has {len(data)}, report says n={n}"
        for row in data:
            assert {"question", "reference", "prediction"}.issubset(row)
