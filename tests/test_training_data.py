"""Data-integrity tests for the distillation QA pairs."""
import json
import os

import pytest

TRAIN = "finetune_data_v2/train.jsonl"
VALID = "finetune_data_v2/valid.jsonl"
REQUIRED_ROLES = {"system", "user", "assistant"}


def _load_jsonl(repo_root, rel):
    path = os.path.join(repo_root, rel)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("rel", [TRAIN, VALID])
def test_jsonl_parseable_and_non_empty(repo_root, rel):
    rows = _load_jsonl(repo_root, rel)
    assert len(rows) > 0, f"{rel} has no examples"


@pytest.mark.parametrize("rel,min_count", [(TRAIN, 1000), (VALID, 200)])
def test_expected_example_count(repo_root, rel, min_count):
    rows = _load_jsonl(repo_root, rel)
    assert len(rows) >= min_count, (
        f"{rel} has {len(rows)} examples, expected at least {min_count}"
    )


@pytest.mark.parametrize("rel", [TRAIN, VALID])
def test_every_row_has_three_roles(repo_root, rel):
    rows = _load_jsonl(repo_root, rel)
    for i, row in enumerate(rows):
        assert "messages" in row, f"{rel}[{i}] missing 'messages'"
        roles = {m["role"] for m in row["messages"]}
        assert REQUIRED_ROLES.issubset(roles), (
            f"{rel}[{i}] roles={roles}, missing {REQUIRED_ROLES - roles}"
        )


@pytest.mark.parametrize("rel", [TRAIN, VALID])
def test_assistant_message_non_empty(repo_root, rel):
    rows = _load_jsonl(repo_root, rel)
    for i, row in enumerate(rows):
        asst = next(m for m in row["messages"] if m["role"] == "assistant")
        assert asst["content"].strip(), f"{rel}[{i}] empty assistant content"
