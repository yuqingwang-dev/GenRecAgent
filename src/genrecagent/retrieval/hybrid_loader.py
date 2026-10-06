"""Reuse verified M1 artifacts and the existing full-trained M3 checkpoint."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from genrecagent.agent.offline_factory import OfflineRouterResources, load_offline_router
from genrecagent.retrieval.generative import GenerativeRetriever, train_popularity_counts
from genrecagent.retrieval.hybrid import DualRetriever
from genrecagent.retrieval.transformer import (
    CausalSemanticTransformer,
    SemanticVocabulary,
    TransformerTokenScorer,
)
from genrecagent.semantic_id.codebook import SemanticCodebook


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class DualResources:
    dual: DualRetriever
    base: OfflineRouterResources
    checkpoint_sha256: str
    m3_manifest: dict


def load_dual_resources(root: Path, *, device: str = "cuda") -> DualResources:
    prepared = root / "data/processed/All_Beauty_full_optimized"
    sequences = root / "data/processed/All_Beauty_m3"
    codes = root / "outputs/m2_rqkmeans_full/item_semantic_ids.json"
    checkpoint = root / "outputs/m3_transformer_full_server/transformer.pt"
    manifest = json.loads((sequences / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((checkpoint.parent / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((checkpoint.parent / "metrics.json").read_text(encoding="utf-8"))
    code_hash = file_sha256(codes)
    if code_hash != manifest["semantic_ids_sha256"] or code_hash != config["semantic_ids_sha256"]:
        raise ValueError("Semantic IDs differ from the full-trained checkpoint")
    if file_sha256(sequences / "train.jsonl") != config["train_examples_sha256"]:
        raise ValueError("Training samples differ from the checkpoint")
    if file_sha256(prepared / "histories.jsonl") != manifest["histories_sha256"]:
        raise ValueError("History source differs from M3 samples")
    if metrics["status"] != "full_train_prefixes":
        raise ValueError("A full-trained checkpoint is required")
    base = load_offline_router(prepared, root / "data/processed/All_Beauty/index")
    book = SemanticCodebook.load(codes)
    model, vocabulary, max_history = CausalSemanticTransformer.load(checkpoint)
    if vocabulary.tokens != SemanticVocabulary.from_codebook(book).tokens:
        raise ValueError("Vocabulary does not match the catalog")
    generative = GenerativeRetriever(
        book, TransformerTokenScorer(model, vocabulary, max_history=max_history, device=device),
        train_popularity=train_popularity_counts(base.histories), beam_size=20, max_codes=20,
    )
    return DualResources(DualRetriever(base.router.dense, generative), base,
                         file_sha256(checkpoint), manifest)
