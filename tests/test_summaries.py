"""Smoke tests for the committed SEC summary JSONs."""
import glob
import json
import os

import pytest


def _summary_paths(repo_root):
    return sorted(glob.glob(os.path.join(repo_root, "summaries", "*_summary.json")))


def test_summaries_present(repo_root):
    paths = _summary_paths(repo_root)
    assert len(paths) >= 20, f"expected ≥20 summary files, found {len(paths)}"


@pytest.mark.parametrize("key", ["top_risks", "strategic_highlights", "mda_summary"])
def test_summaries_have_required_keys(repo_root, key):
    for path in _summary_paths(repo_root):
        data = json.load(open(path))
        assert key in data, f"{os.path.basename(path)} missing '{key}'"
