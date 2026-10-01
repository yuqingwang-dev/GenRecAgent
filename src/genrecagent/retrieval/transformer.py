"""Small causal Transformer for next-item Semantic ID tokens."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from genrecagent.retrieval.semantic_trie import HistoryCodes, SemanticID
from genrecagent.semantic_id.codebook import SemanticCodebook


class SemanticVocabulary:
    """One fixed token vocabulary plus three special sequence markers."""

    def __init__(self, tokens: tuple[str, ...]) -> None:
        if not tokens or len(tokens) != len(set(tokens)):
            raise ValueError("Semantic vocabulary must contain unique tokens")
        self.tokens = tokens
        self.to_id = {token: index for index, token in enumerate(tokens)}
        self.pad_id = len(tokens)
        self.item_end_id = len(tokens) + 1
        self.predict_id = len(tokens) + 2
        self.size = len(tokens) + 3

    @classmethod
    def from_codebook(cls, codebook: SemanticCodebook) -> "SemanticVocabulary":
        tokens = tuple(sorted({token for code in codebook.assignments.values() for token in code}))
        return cls(tokens)

    def history_ids(self, history: HistoryCodes, *, max_history: int) -> list[int]:
        if max_history < 1:
            raise ValueError("max_history must be positive")
        return [
            part
            for code in history[-max_history:]
            for part in (*[self.to_id[token] for token in code], self.item_end_id)
        ]

    def teacher_forcing(
        self, history: HistoryCodes, target: SemanticID, *, max_history: int
    ) -> tuple[list[int], list[int]]:
        if len(target) != 3:
            raise ValueError("This M3 model expects three target tokens")
        prefix = self.history_ids(history, max_history=max_history)
        target_ids = [self.to_id[token] for token in target]
        inputs = [*prefix, self.predict_id, *target_ids[:-1]]
        labels = [-100] * len(prefix) + target_ids
        return inputs, labels


def collate_teacher_forced(
    examples: list[dict[str, object]],
    vocabulary: SemanticVocabulary,
    *,
    max_history: int,
) -> tuple[Tensor, Tensor]:
    encoded = [
        vocabulary.teacher_forcing(
            tuple(tuple(code) for code in row["history_codes"]),
            tuple(row["target_code"]),
            max_history=max_history,
        )
        for row in examples
    ]
    length = max(len(inputs) for inputs, _ in encoded)
    input_ids = torch.full((len(encoded), length), vocabulary.pad_id, dtype=torch.long)
    labels = torch.full((len(encoded), length), -100, dtype=torch.long)
    for index, (inputs, targets) in enumerate(encoded):
        input_ids[index, :len(inputs)] = torch.tensor(inputs)
        labels[index, :len(targets)] = torch.tensor(targets)
    return input_ids, labels


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    vocab_size: int
    max_positions: int
    pad_id: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if min(
            self.vocab_size, self.max_positions, self.d_model,
            self.n_heads, self.n_layers, self.feedforward_dim,
        ) < 1 or self.d_model % self.n_heads:
            raise ValueError("Transformer dimensions and head count are incompatible")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


class CausalSemanticTransformer(nn.Module):
    """Predict only target code tokens from history and previous target tokens."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_positions, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers,
                                             enable_nested_tensor=False)
        self.output = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.max_positions:
            raise ValueError("Input must be [batch, sequence] within max_positions")
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=input_ids.device), diagonal=1
        )
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=input_ids.eq(self.config.pad_id),
        )
        return self.output(hidden)

    def target_loss(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        logits = self(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), labels.reshape(-1), ignore_index=-100
        )

    def save(self, path: Path, vocabulary: SemanticVocabulary, max_history: int) -> None:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": asdict(self.config), "tokens": vocabulary.tokens,
            "max_history": max_history, "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(
        cls, path: Path, *, map_location: str = "cpu"
    ) -> tuple["CausalSemanticTransformer", SemanticVocabulary, int]:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        vocabulary = SemanticVocabulary(tuple(payload["tokens"]))
        model = cls(TransformerConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model, vocabulary, int(payload["max_history"])


class TransformerTokenScorer:
    """Batched log-probability adapter for catalog-constrained beam search."""

    def __init__(
        self,
        model: CausalSemanticTransformer,
        vocabulary: SemanticVocabulary,
        *,
        max_history: int,
        device: str,
    ) -> None:
        self.model = model.to(device).eval()
        self.vocabulary = vocabulary
        self.max_history = max_history
        self.device = device

    @torch.no_grad()
    def log_probabilities_batch(
        self, history: HistoryCodes, prefixes: tuple[SemanticID, ...]
    ) -> list[dict[str, float]]:
        context = self.vocabulary.history_ids(history, max_history=self.max_history)
        sequences = [
            [*context, self.vocabulary.predict_id,
             *[self.vocabulary.to_id[token] for token in prefix]]
            for prefix in prefixes
        ]
        length = max(map(len, sequences))
        inputs = torch.full(
            (len(sequences), length), self.vocabulary.pad_id, dtype=torch.long, device=self.device
        )
        for index, sequence in enumerate(sequences):
            inputs[index, :len(sequence)] = torch.tensor(sequence, device=self.device)
        logits = self.model(inputs)
        final_logits = torch.stack([
            logits[index, len(sequence) - 1] for index, sequence in enumerate(sequences)
        ])
        log_probs = F.log_softmax(final_logits, dim=-1).cpu().numpy()
        return [
            {token: float(row[index]) for index, token in enumerate(self.vocabulary.tokens)}
            for row in log_probs
        ]

    def log_probabilities(
        self, history: HistoryCodes, prefix: SemanticID
    ) -> dict[str, float]:
        return self.log_probabilities_batch(history, (prefix,))[0]
