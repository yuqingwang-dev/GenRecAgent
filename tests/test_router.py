from genrecagent.agent.router import Router


def test_router_maps_core_actions() -> None:
    router = Router()

    assert router.route_plan(["filter", "retrieval", "ranking"]) == [
        "filter",
        "retrieval",
        "ranking",
    ]
    assert router.route("unknown") is None

