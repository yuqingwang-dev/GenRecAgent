"""Dynamic plan-action to tool-name routing."""


class Router:
    """Map logical plan actions to registered tool names."""

    def __init__(self, routes: dict[str, str] | None = None) -> None:
        self.routes = routes or {
            "filter": "filter",
            "retrieval": "retrieval",
            "ranking": "ranking",
        }

    def route(self, action: str) -> str | None:
        return self.routes.get(action)

    def route_plan(self, plan: list[str]) -> list[str | None]:
        return [self.route(action) for action in plan]

