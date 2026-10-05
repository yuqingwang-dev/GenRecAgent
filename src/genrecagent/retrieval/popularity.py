"""Popularity ranking from training interactions only."""

from collections import Counter
from itertools import islice

from genrecagent.data.models import UserHistory
from genrecagent.retrieval.dense import SearchHit


class PopularityRetriever:
    def __init__(self, histories: list[UserHistory], item_ids: list[str]) -> None:
        counts = Counter(item_id for user in histories for item_id in user.train_items)
        self.ranked = sorted(item_ids, key=lambda item_id: (-counts[item_id], item_id))
        self.counts = counts

    def search(
        self, *, top_k: int,
        exclude_seen_items: set[str] | frozenset[str] | None = None,
        allowed_item_ids: set[str] | frozenset[str] | None = None,
    ) -> list[SearchHit]:
        seen = exclude_seen_items or set()
        eligible = (
            item_id for item_id in self.ranked if item_id not in seen
            and (allowed_item_ids is None or item_id in allowed_item_ids)
        )
        return [
            SearchHit(
                item_id=item_id,
                score=float(self.counts[item_id]),
                rank=index,
                source="popularity",
            )
            for index, item_id in enumerate(islice(eligible, max(0, top_k)), start=1)
        ]
