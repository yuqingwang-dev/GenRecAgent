import json

from genrecagent.agent.state import AgentState, UserRequest


def test_agent_state_is_json_serializable() -> None:
    state = AgentState(
        original_request=UserRequest(
            query="running shoes",
            user_id="u1",
            user_history=("fitness",),
            constraints={"max_price": 100},
        )
    )

    payload = state.to_dict()

    assert payload["original_request"]["user_id"] == "u1"
    json.dumps(payload)

