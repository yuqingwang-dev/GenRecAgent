"""Disable a physical source for experiments without changing the user's requested route."""

from genrecagent.agent.tool_environment import ToolEnvironment


class SourceAblationEnvironment(ToolEnvironment):
    def __init__(self, resources, items, task, *, disabled_tool: str):
        if disabled_tool not in {"retrieve_dense", "retrieve_generative"}:
            raise ValueError("Only real retrieval-source removal is supported")
        super().__init__(resources, items, task)
        self.disabled_tool = disabled_tool

    def _execute(self, name, args):
        if name == self.disabled_tool:
            # Validate the normal prerequisites first, without running the disabled retriever.
            if not self.history_read:
                raise ValueError("Read history first")
            count = args.get("top_k", 50)
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 100:
                raise ValueError("top_k must be an integer in [1,100]")
            raise RuntimeError("source_disabled_for_ablation")
        return super()._execute(name, args)
