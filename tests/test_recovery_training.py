import json

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

from genrecagent.agent.action_loss import assistant_action_loss
from genrecagent.agent.compact_policy import compact_state
from genrecagent.agent.recovery_data import balanced_actions, check_split_groups
from genrecagent.agent.rollout_validation import selection_score, validation_subset
from genrecagent.agent.tool_environment import TOOL_NAMES
from genrecagent.agent.trajectory_data import action_tokens
from genrecagent.llm.local_qwen import LocalQwenLLM


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


def test_rollout_selection_is_validation_only_and_prioritizes_feasible_fills():
    rows = [
        {
            "task_id": f"v{i}",
            "split": "validation",
            "expected": "success" if i < 4 else "clarification",
        }
        for i in range(8)
    ]
    chosen = validation_subset(rows, count=4)
    assert sum(row["expected"] == "success" for row in chosen) == 2
    assert chosen == validation_subset(rows, count=4)
    with pytest.raises(ValueError, match="only"):
        validation_subset([{**rows[0], "split": "test"}], count=2)
    strong = {"feasible_filled": 2, "contract_pass": 3, "policy_illegal_calls": 0}
    low_loss_but_empty = {"feasible_filled": 0, "contract_pass": 4, "policy_illegal_calls": 0}
    assert selection_score(strong, 1.0) > selection_score(low_loss_but_empty, 0.001)


def test_loaded_trainable_adapter_can_generate_between_training_epochs():
    base = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model = get_peft_model(
        base,
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.eval()
    model.disable_input_require_grads()

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2

        def apply_chat_template(self, messages, **kwargs):
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([[1, 3, 4]]),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long),
                }
            )

        def decode(self, suffix, **kwargs):
            return "{}"

    llm = LocalQwenLLM.from_loaded_model(
        model, Tokenizer(), device="cpu", model_name="tiny-software", revision="a" * 40
    )
    llm.generation_config.max_new_tokens = 2
    assert llm.chat([{"role": "user", "content": "state"}]) == "{}"
    assert llm.generation_record()["output_tokens"] <= 2
    model.enable_input_require_grads()
    model.train()
    row = {"input_ids": [1, 3, 4], "labels": [-100, 3, 4]}
    loss = assistant_action_loss(model, row, "cpu")
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
