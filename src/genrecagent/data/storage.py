"""JSONL persistence and explicit Parquet loading for the M1 dataset."""

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, TypeVar

from genrecagent.data.amazon import PreparedData
from genrecagent.data.models import Interaction, Item, UserHistory

Record = TypeVar("Record", Item, Interaction, UserHistory)


def iter_parquet_rows(path: Path, columns: list[str]) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Install pyarrow in the genrecagent environment to read Parquet"
        ) from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        yield from batch.to_pylist()


def write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _history(row: Mapping[str, Any]) -> UserHistory:
    return UserHistory(
        user_id=row["user_id"],
        train_items=tuple(row["train_items"]),
        validation_item=row["validation_item"],
        test_item=row["test_item"],
        train_timestamps=tuple(row["train_timestamps"]),
        validation_timestamp=row["validation_timestamp"],
        test_timestamp=row["test_timestamp"],
    )


def save_prepared(data: PreparedData, path: Path, manifest: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_jsonl(path / "items.jsonl", (data.items[key] for key in sorted(data.items)))
    write_jsonl(path / "interactions.jsonl", data.interactions)
    write_jsonl(path / "histories.jsonl", data.histories)
    (path / "manifest.json").write_text(
        json.dumps({**manifest, "stats": data.stats}, indent=2), encoding="utf-8"
    )


def load_items(path: Path) -> dict[str, Item]:
    return {
        row["item_id"]: Item(
            item_id=row["item_id"],
            title=row["title"],
            description=row["description"],
            features=tuple(row["features"]),
            categories=tuple(row["categories"]),
            price=row["price"],
            average_rating=row["average_rating"],
        )
        for row in read_jsonl(path / "items.jsonl")
    }


def load_histories(path: Path) -> list[UserHistory]:
    return [_history(row) for row in read_jsonl(path / "histories.jsonl")]
