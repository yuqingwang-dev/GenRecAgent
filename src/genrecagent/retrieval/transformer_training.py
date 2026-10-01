"""Train a small next-code Transformer using train prefixes and validation labels."""

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from genrecagent.retrieval.transformer import (
    CausalSemanticTransformer,
    SemanticVocabulary,
    TransformerConfig,
    collate_teacher_forced,
)


@dataclass(frozen=True, slots=True)
class TransformerTrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 5e-4
    seed: int = 42
    device: str = "auto"
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


@dataclass(slots=True)
class TransformerTrainingResult:
    model: CausalSemanticTransformer
    best_epoch: int
    best_validation_loss: float
    epoch_log: list[dict[str, float | int]]
    device: str


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def _check_splits(train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> None:
    if not train_rows or not validation_rows:
        raise ValueError("Training and validation examples must be nonempty")
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("Only train-prefix examples may be used for training")
    if any(row.get("split") != "validation" for row in validation_rows):
        raise ValueError("Checkpoint selection must use validation examples")


def fit_semantic_transformer(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    vocabulary: SemanticVocabulary,
    model_config: TransformerConfig,
    train_config: TransformerTrainingConfig,
    *,
    max_history: int,
) -> TransformerTrainingResult:
    _check_splits(train_rows, validation_rows)
    if model_config.max_positions < max_history * 4 + 3:
        raise ValueError("max_positions is too small for max_history")
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)
    device = _device(train_config.device)
    model = CausalSemanticTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    rng = np.random.default_rng(train_config.seed)
    best_validation = math.inf
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    log: list[dict[str, float | int]] = []
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        train_total = 0.0
        for indices in np.array_split(
            rng.permutation(len(train_rows)),
            max(1, math.ceil(len(train_rows) / train_config.batch_size)),
        ):
            rows = [train_rows[int(index)] for index in indices]
            input_ids, labels = collate_teacher_forced(
                rows, vocabulary, max_history=max_history
            )
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.target_loss(input_ids, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
            optimizer.step()
            train_total += float(loss.detach()) * len(rows)
        model.eval()
        validation_total = 0.0
        correct_tokens = total_tokens = 0
        with torch.no_grad():
            for start in range(0, len(validation_rows), train_config.batch_size):
                rows = validation_rows[start:start + train_config.batch_size]
                input_ids, labels = collate_teacher_forced(
                    rows, vocabulary, max_history=max_history
                )
                input_ids, labels = input_ids.to(device), labels.to(device)
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, model_config.vocab_size),
                    labels.reshape(-1), ignore_index=-100,
                )
                validation_total += float(loss) * len(rows)
                supervised = labels.ne(-100)
                correct_tokens += int((logits.argmax(dim=-1).eq(labels) & supervised).sum())
                total_tokens += int(supervised.sum())
        validation_loss = validation_total / len(validation_rows)
        if not math.isfinite(validation_loss):
            raise RuntimeError("Non-finite validation loss")
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_target_nll": train_total / len(train_rows),
            "validation_target_nll": validation_loss,
            "validation_token_accuracy": correct_tokens / total_tokens,
        }
        log.append(record)
        print(
            f"epoch={epoch} train_target_nll={record['train_target_nll']:.5f} "
            f"val_target_nll={validation_loss:.5f}",
            flush=True,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return TransformerTrainingResult(model, best_epoch, best_validation, log, device)
