import pytest

from genrecagent.agent.trajectory_data import action_tokens


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "prefix" if add_generation_prompt else "prefixaction"

    def encode(self, text, *, add_special_tokens):
        return [1, 2, 3] if text == "prefix" else [40, 41]


def test_only_assistant_action_tokens_are_supervised() -> None:
    rows = [{"role": "user", "content": "state"}, {"role": "assistant", "content": "action"}]
    result = action_tokens(Tokenizer(), rows)
    assert result["labels"] == [-100, -100, -100, 40, 41]
    with pytest.raises(ValueError, match="silently truncate"):
        action_tokens(Tokenizer(), rows, max_length=4)
    with pytest.raises(ValueError, match="assistant"):
        action_tokens(Tokenizer(), rows[:-1])
