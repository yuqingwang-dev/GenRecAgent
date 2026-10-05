"""Strict tool JSON decoding: reject duplicate keys, nonfinite values and prose."""

import json
from typing import Any


class ToolJSONError(ValueError):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ToolJSONError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ToolJSONError(f"Nonfinite JSON value: {value}")


def decode_tool_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ToolJSONError("Response must be exactly one JSON object") from exc
    if not isinstance(payload, dict):
        raise ToolJSONError("Tool response must be an object")
    return payload
