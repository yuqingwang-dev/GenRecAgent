"""Assistant-action supervision with user/state/tool-observation tokens masked."""

import json

from genrecagent.agent.tool_environment import STEP_PROMPT, TOOL_SCHEMAS
from genrecagent.llm.tool_json import tool_messages


def step_messages(state: dict, action: dict | None = None) -> list[dict[str, str]]:
    messages = tool_messages(
        STEP_PROMPT + "\nSTATE=" + json.dumps(state, ensure_ascii=False), TOOL_SCHEMAS
    )
    if action is not None:
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
    return messages


def action_tokens(
    tokenizer, messages: list[dict], *, max_length: int = 1024, action_name_weight: float = 1.0
) -> dict:
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("A target assistant action is required")
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not full_text.startswith(prefix_text) or len(full_text) <= len(prefix_text):
        raise ValueError("Chat template does not preserve the assistant prefix")
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    target = tokenizer.encode(full_text[len(prefix_text) :], add_special_tokens=False)
    full = prefix + target
    if len(full) > max_length:
        raise ValueError(f"Action sample exceeds {max_length} tokens; do not silently truncate")
    result = {"input_ids": full, "labels": [-100] * len(prefix) + target}
    if action_name_weight != 1.0:
        if not 1 <= action_name_weight <= 10:
            raise ValueError("Action name weight must be in [1,10]")
        action = json.loads(messages[-1]["content"])
        needle = json.dumps(action["name"], ensure_ascii=False)
        action_text = messages[-1]["content"]
        start = action_text.index(needle, action_text.index('"name"') + len('"name"')) + 1
        end = start + len(action["name"])
        continuation = full_text[len(prefix_text) :]
        encoded = tokenizer(continuation, add_special_tokens=False, return_offsets_mapping=True)
        if encoded["input_ids"] != target:
            raise ValueError("Offset encoding differs from target encoding")
        weights = [
            action_name_weight if left < end and right > start else 1.0
            for left, right in encoded["offset_mapping"]
        ]
        if action_name_weight not in weights:
            raise ValueError("No action-name target tokens were found")
        result["loss_weights"] = [0.0] * len(prefix) + weights
    return result
