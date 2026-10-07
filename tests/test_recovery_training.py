import json

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from genrecagent.agent.action_loss import assistant_action_loss
from genrecagent.agent.compact_policy import compact_state
from genrecagent.agent.recovery_data import balanced_actions, check_split_groups
from genrecagent.agent.tool_environment import TOOL_NAMES
from genrecagent.agent.trajectory_data import action_tokens


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "PREFIX" if add_generation_prompt else "PREFIX" + messages[-1]["content"] + "END"

    def encode(self, text, *, add_special_tokens):
        return [ord(char) for char in text]

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text, add_special_tokens=False),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def test_name_weighting_masks_prefix_and_only_upweights_name_span():
    messages = [
        {"role": "user", "content": "observable state"},
        {"role": "assistant", "content": json.dumps({"name": "finish", "arguments": {}})},
    ]
    encoded = action_tokens(CharacterTokenizer(), messages, action_name_weight=4)
    assert encoded["labels"][:6] == [-100] * 6
    assert encoded["loss_weights"][:6] == [0] * 6
    assert encoded["loss_weights"].count(4) == len("finish")
    assert all(weight in {0, 1, 4} for weight in encoded["loss_weights"])


def test_all_one_weighted_causal_loss_matches_original_and_rejects_prefix_supervision():
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    ).eval()
    row = {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]}
    original = assistant_action_loss(model, row, "cpu")
    weighted = assistant_action_loss(model, {**row, "loss_weights": [0, 0, 1, 1, 1]}, "cpu")
    torch.testing.assert_close(original, weighted)
    with pytest.raises(ValueError, match="mask"):
        assistant_action_loss(model, {**row, "loss_weights": [1, 0, 1, 1, 1]}, "cpu")


def test_balancing_only_uses_executed_training_examples_and_reports_duplicates():
    rows = [
        {
            "split": "train",
            "executed_action_checked": True,
            "sample_id": name,
            "action": {"name": name},
        }
        for name in TOOL_NAMES
    ]
    selected, report = balanced_actions(rows, per_action=3)
    assert len(selected) == 27 and report["distinct_selected_actions"] == 9
    assert set(report["selected_actions"].values()) == {3}
    with pytest.raises(ValueError, match="training"):
        balanced_actions([{**rows[0], "split": "validation"}])


def test_recovery_user_and_goal_groups_must_be_disjoint():
    def row(user, price):
        return {
            "user_group": user,
            "mode": "dense",
            "max_price": price,
            "category": None,
            "top_k": 5,
        }

    groups = {"train": [row("train", 1)], "validation": [row("val", 2)], "test": [row("test", 3)]}
    check_split_groups(groups)
    groups["test"] = [row("train", 3)]
    with pytest.raises(ValueError, match="leakage"):
        check_split_groups(groups)


def test_compact_state_removes_item_payloads_without_adding_action_or_phase_oracle():
    state = {
        "request": {"mode": "dense"},
        "history_read": False,
        "observed_history": None,
        "source_counts": {},
        "source_errors": {},
        "pending_sources": ["dense"],
        "eligible_count": 2,
        "filtered": False,
        "ranked_ready": False,
        "ranked_count": 0,
        "verified": False,
        "sufficient": False,
        "steps_left": 10,
        "last_observation": {"ok": False, "error": "Read history first", "items": ["PRIVATE"]},
    }
    compact = compact_state(state)
    assert compact["history_observed_sample_size"] is None
    assert "items" not in compact["last_observation"]
    assert not ({"phase", "allowed_actions", "next_action", "expected"} & compact.keys())
    assert compact["last_observation"]["error"] == "Read history first"
