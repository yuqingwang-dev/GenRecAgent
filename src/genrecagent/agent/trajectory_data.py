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


def action_tokens(tokenizer, messages: list[dict], *, max_length: int = 1024) -> dict:
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
    return {"input_ids": full, "labels": [-100] * len(prefix) + target}
