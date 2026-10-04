"""Load a real-data history router without remote models or test-target training."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genrecagent.agent.offline_router import OfflineRecommendationRouter
from genrecagent.data.models import UserHistory
from genrecagent.data.storage import load_histories, read_jsonl
from genrecagent.retrieval.dense import DenseRetriever
from genrecagent.retrieval.generative import GenerativeRetriever, train_popularity_counts
from genrecagent.retrieval.popularity import PopularityRetriever
from genrecagent.retrieval.transition import CodeTransitionScorer
from genrecagent.semantic_id.codebook import SemanticCodebook


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OfflineRouterResources:
    router: OfflineRecommendationRouter
    histories: tuple[UserHistory, ...]
    prepared_manifest: dict[str, Any]
    m3_manifest: dict[str, Any] | None
    train_examples_sha256: str | None


def load_offline_router(
    prepared_dir: Path,
    index_dir: Path,
    *,
    enable_generative: bool = False,
    m3_sequences_dir: Path | None = None,
    semantic_ids_path: Path | None = None,
    transition_alpha: float = 10.0,
    beam_size: int = 20,
    max_codes: int = 20,
) -> OfflineRouterResources:
    """Use M1 index and train-only popularity; load M3 only if explicitly enabled."""

    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    dense = DenseRetriever.load_index(index_dir)
    if manifest["metadata_sha256"] != dense.metadata["metadata_sha256"]:
        raise ValueError("Dense index metadata hash differs from the prepared catalog")
    histories = tuple(load_histories(prepared_dir))
    popularity = PopularityRetriever(list(histories), dense.item_ids)
    m3_manifest = None
    train_hash = None
    generative = None
    if enable_generative:
        if m3_sequences_dir is None or semantic_ids_path is None:
            raise ValueError("Generative route needs M3 sequences and Semantic IDs")
        m3_manifest = json.loads(
            (m3_sequences_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if _sha256(prepared_dir / "histories.jsonl") != m3_manifest["histories_sha256"]:
            raise ValueError("M3 and M1 histories differ")
        if _sha256(semantic_ids_path) != m3_manifest["semantic_ids_sha256"]:
            raise ValueError("Semantic ID mapping differs from M3 examples")
        train_file = m3_sequences_dir / "train.jsonl"
        train_hash = _sha256(train_file)
        codebook = SemanticCodebook.load(semantic_ids_path)
        scorer = CodeTransitionScorer(codebook, alpha=transition_alpha).fit(
            read_jsonl(train_file)
        )
        generative = GenerativeRetriever(
            codebook, scorer, train_popularity=train_popularity_counts(histories),
            beam_size=beam_size, max_codes=max_codes,
        )
    return OfflineRouterResources(
        router=OfflineRecommendationRouter(dense, popularity, generative),
        histories=histories,
        prepared_manifest=manifest,
        m3_manifest=m3_manifest,
        train_examples_sha256=train_hash,
    )
