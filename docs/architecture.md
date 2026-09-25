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

## M1 recommendation path (in progress)

```text
Amazon Reviews 2023 / All_Beauty
  reviews -> parent_asin interactions -> per-user ordered train/validation/test
  metadata -> item text -> BGE encoder -> normalized item embeddings
  train history item embeddings -> mean pool -> normalized user embedding
  user embedding -> exact inner-product index -> unseen top-K items
  train interactions -> popularity counts -> unseen top-K items
  both methods -> same leave-one-out Recall/NDCG evaluator
```

The NumPy exact backend has been exercised locally with deterministic vectors. The optional FAISS backend uses `IndexFlatIP`, not an approximate index. Item IDs and metadata are persisted next to the vectors; reloading is checked in a fixture test. The BGE and official Parquet paths are implemented but have not run on real artifacts yet.

Review timestamps establish within-user order only. The catalog is treated as a static offline universe. Train-only popularity is safe under that protocol, while cross-user global temporal separation and historical catalog availability are not guaranteed. Semantic ID, generative retrieval and learned agent policy remain future components and are not connected to this M1 path.
