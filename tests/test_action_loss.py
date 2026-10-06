import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from genrecagent.agent.action_loss import assistant_action_loss


def test_action_logits_match_full_causal_loss_and_gradients() -> None:
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_dropout=0.0,
        )
    ).eval()
    row = {"input_ids": [1, 2, 3, 4, 5, 6], "labels": [-100, -100, -100, 4, 5, 6]}
    full = model(
        input_ids=torch.tensor([row["input_ids"]]),
        labels=torch.tensor([row["labels"]]),
        use_cache=False,
    ).loss
    full.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad()
    reduced = assistant_action_loss(model, row, "cpu")
    torch.testing.assert_close(reduced, full)
    reduced.backward()
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[name], atol=1e-6, rtol=1e-4)


def test_action_logits_reject_missing_prefix_or_noncontiguous_targets() -> None:
    for labels in ([1, 2], [-100, -100], [-100, 2, -100]):
        with pytest.raises(ValueError):
            assistant_action_loss(None, {"input_ids": [1] * len(labels), "labels": labels}, "cpu")
