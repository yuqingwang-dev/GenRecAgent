from genrecagent.data.amazon import build_item_text, normalize_item, prepare_records


def _meta(item_id: str, **overrides: object) -> dict[str, object]:
    return {"parent_asin": item_id, "title": item_id, **overrides}


def _review(user: str, item_id: str, timestamp: int) -> dict[str, object]:
    return {"user_id": user, "parent_asin": item_id, "timestamp": timestamp}


def test_chronological_split_deduplicates_and_filters_missing_metadata() -> None:
    meta = [_meta(key) for key in ("A", "B", "C", "D", "E")]
    reviews = [
        _review("u", "E", 50),
        _review("u", "C", 30),
        _review("u", "A", 10),
        _review("u", "D", 40),
        _review("u", "B", 20),
        _review("u", "A", 60),
        _review("u", "missing", 70),
        _review("short", "A", 1),
    ]
    data = prepare_records(meta, reviews, seed=7)
    history = data.histories[0]
    assert history.train_items == ("A", "B", "C")
    assert history.validation_item == "D"
    assert history.test_item == "E"
    assert history.test_history == ("A", "B", "C", "D")
    assert max(history.train_timestamps) < history.validation_timestamp
    assert history.validation_timestamp < history.test_timestamp
    assert data.stats["duplicate_user_item_reviews"] == 1
    assert data.stats["reviews_without_metadata"] == 1
    assert data.stats["users_below_min_interactions"] == 1


def test_equal_timestamps_use_source_order_and_user_sample_is_stable() -> None:
    meta = [_meta(key) for key in ("A", "B", "C")]
    reviews = [
        _review(user, key, 100) for user in ("u1", "u2", "u3") for key in ("A", "B", "C")
    ]
    first = prepare_records(meta, reviews, max_users=2, seed=99)
    second = prepare_records(meta, reversed(reviews), max_users=2, seed=99)
    assert {h.user_id for h in first.histories} == {h.user_id for h in second.histories}
    assert first.stats["equal_timestamp_adjacent_pairs"] == 6
    assert all(h.train_items == ("A",) for h in first.histories)


def test_item_text_tolerates_missing_and_unusual_metadata() -> None:
    item = normalize_item(
        {
            "parent_asin": "X",
            "title": None,
            "features": [None, "Hydrating", {}, ""],
            "categories": None,
            "description": ["For dry skin", None],
            "price": "unknown",
        }
    )
    assert item is not None
    assert item.price is None
    assert build_item_text(item) == "Features: Hydrating\nDescription: For dry skin"
