"""Exact action-only causal loss without projecting masked prompt logits."""

import torch
from torch.nn import functional as F


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
    inputs = dict(
        input_ids=torch.tensor([row["input_ids"]], device=device),
        labels=torch.tensor([labels[-keep:]], device=device),
        logits_to_keep=keep,
        use_cache=False,
    )
    if "loss_weights" not in row:
        return model(**inputs).loss
    weights = row["loss_weights"]
    if len(weights) != len(labels) or any(weight != 0 for weight in weights[:first]):
        raise ValueError("Loss weights must align with labels and mask the context")
    if not all(0 < weight <= 10 for weight in weights[first:]):
        raise ValueError("Target weights must be finite positive values in (0,10]")
    targets = inputs.pop("labels")
    logits = model(**inputs).logits.float()
    shifted = F.pad(targets, (0, 1), value=-100)[:, 1:]
    token_weights = torch.tensor([weights[-keep:][1:] + [0.0]], device=device)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shifted.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(token_weights)
    return (losses * token_weights).sum() / token_weights.sum()
