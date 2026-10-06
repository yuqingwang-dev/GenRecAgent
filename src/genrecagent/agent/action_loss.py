"""Exact action-only causal loss without projecting masked prompt logits."""

import torch


def assistant_action_loss(model, row: dict, device: str):
    labels = row["labels"]
    positions = [index for index, label in enumerate(labels) if label != -100]
    if not positions or positions[0] == 0:
        raise ValueError("Action labels need a masked context prefix")
    first = positions[0]
    if positions != list(range(first, len(labels))) or len(labels) != len(row["input_ids"]):
        raise ValueError("Expected contiguous assistant labels after a masked context prefix")
    # Retain the last context position: its logits predict the FIRST action token.
    keep = len(labels) - first + 1
    return model(
        input_ids=torch.tensor([row["input_ids"]], device=device),
        labels=torch.tensor([labels[-keep:]], device=device),
        logits_to_keep=keep,
        use_cache=False,
    ).loss
