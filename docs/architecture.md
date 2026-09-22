# Architecture Notes

## Phase 0 contracts

`UserRequest` is the boundary object for query, user identity, optional history, and constraints. `AgentState` records the original request, plan, selected tools, tool results, candidates, verification, bounded step count, completion, and final response. Each state component can be converted to a JSON-compatible dictionary for trace storage.

`BaseTool` is the only contract the executor needs. This keeps tool selection independent from implementation details such as an in-memory catalog, a dense index, a semantic-ID generator, or an external service.

## Control flow

The Phase 0 agent is deterministic:

1. `Planner` creates an intent-independent baseline plan.
2. `Router` maps plan actions to registered tools.
3. `Executor` runs one tool and records a `ToolResult` with latency.
4. `Verifier` checks that the result contains candidates and satisfies a known price constraint.
5. A failed run may request one re-plan, but `max_steps` always bounds execution.

The control loop does not call an LLM. The LLM interface is isolated so provider-specific code can be introduced later without changing state or tool contracts.

