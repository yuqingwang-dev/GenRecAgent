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

## M1 recommendation path (full eligible-user evaluation verified)

```text
Amazon Reviews 2023 / All_Beauty
  reviews -> parent_asin interactions -> per-user ordered train/validation/test
  metadata -> item text -> BGE encoder -> normalized item embeddings
  train history item embeddings -> mean pool -> normalized user embedding
  user embedding -> exact inner-product index -> unseen top-K items
  train interactions -> popularity counts -> unseen top-K items
  both methods -> same leave-one-out Recall/NDCG evaluator
```

The NumPy exact backend has been exercised with deterministic vectors and the real All_Beauty catalog. The optional FAISS backend uses `IndexFlatIP`, not an approximate index, but was not installed or measured in this run. Item IDs and metadata are persisted next to the vectors; reloading is checked in a fixture test and in the real evaluation. BGE encoding and direct official JSONL/Parquet ingestion ran locally; the primary real metrics are recorded in [results_m1_full.md](results_m1_full.md), with a separate development subset in [results_m1.md](results_m1.md).

Review timestamps establish within-user order only. The catalog is treated as a static offline universe. Train-only popularity is safe under that protocol, while cross-user global temporal separation and historical catalog availability are not guaranteed. Semantic ID training is a separate M2 path; the actual policy fine-tuning path is described below.

## M2 item Semantic ID path (full-catalog comparison verified)

```text
M1 normalized item vectors -> MLP encoder -> residual quantization (learned levels)
                                                -> MLP decoder -> reconstruction loss
                           -> trained code sequence -> catalog item mapping
```

The quantizer's nearest-code choice is discrete. Straight-through gradients train the encoder through the decoder while a codebook loss and commitment loss update the latent space and selected learned codes. Codebooks are initialized from train-only item latents. A fixed item-level validation subset selects the checkpoint by reconstruction error; the user-sequence test targets are not used. The mapping preserves all item IDs sharing a code. Full-catalog RQ-VAE training ran, and its severe codebook collapse is recorded in [results_m2.md](results_m2.md).

The subsequent full-catalog RQ-VAE run showed severe codebook collapse. M2 also includes an independent tokenizer path: train-only PCA whitening of the same 384-D vectors to 64-D, then three residual MiniBatch K-Means codebooks. Projection and codebooks are saved separately and reloaded for encoding. Its remaining collisions are retained in the same `SemanticCodebook` lookup contract. These RQ-KMeans codes are used by the full M3 Transformer and transition experiments. See [results_m2.md](results_m2.md).

## M3 sequence and candidate boundary

Chronological train histories become prefix-to-next-item-code examples; validation uses train history, and test uses train plus the observed validation interaction. A `NextTokenScorer` supplies log-probabilities to a catalog-valid code trie. `GenerativeRetriever` expands valid codes through all colliding catalog items, orders within groups by train-only popularity, excludes seen items and emits source-labeled candidates with diagnostic counters. Both the train-only count scorer and full-trained Transformer have measured retrieval results. [m3_protocol.md](m3_protocol.md) records the exact contract and leakage boundaries.

Two scorer implementations exist: a train-only count transition baseline ([results_m3_transition.md](results_m3_transition.md)) and a fully trained small causal Transformer ([results_m3_transformer.md](results_m3_transformer.md)). For teacher forcing, history tokens and padding have no supervised loss; a prediction marker and preceding target tokens predict exactly the next item's three Semantic ID tokens. A causal attention mask prevents future-target access. At inference a batched scorer supplies log-probabilities for each beam prefix; the same catalog trie restricts outputs to valid codes. Both routes underperform Dense, so M4 selects Dense by default.

## M4 intent, eligibility and output boundary

`ParsedIntent` carries a fixed-grammar intent and optional `PriceLimit`; unrecognized or conflicting conditions receive an explicit status. `ConstrainedRecommendationAgent` obtains eligible catalog IDs from raw item prices and passes that immutable set into every retrieval fallback. Dense/Popularity search inside the eligible set; generated candidates are verified and filtered before being accepted or triggering fallback. The final output reports verified metadata prices and distinguishes requested-count fulfillment from safe partial/empty results. Missing prices remain allowed only without a budget.

The curated development/held-out task specification is separate from the parser. Its grader checks manual labels and raw item metadata without calling the agent's eligibility helper. [results_m4_constraints.md](results_m4_constraints.md) records the controlled-language scope and evidence.

`LocalQwenLLM` implements the provider-neutral model interface using an immutable cached snapshot and bounded greedy generation. `LLMToolPolicy` checks one declared JSON function schema, grounds budget parameters in the input, applies known capability/route checks, and allows one repair before rule fallback. It supplies a typed `ParsedIntent` to the existing execution layer; the model does not produce final catalog items. Raw first decisions and repaired/guarded results are evaluated separately, and independent grading time is excluded from request timing. [results_m4_llm_policy.md](results_m4_llm_policy.md) records the actual local model run and the remaining raw-policy failures.

## Current real-tool action-policy and SFT boundary

`ToolEnvironment` owns the goal, catalog eligibility, retrieved source pools, immutable observable snapshots, rank/verification state and execution budget. The policy proposes only a checked JSON action; it cannot change the goal or invent final products. `FixedStepPolicy` is deterministic state-aware orchestration. `QwenStepPolicy` runs either the base model or its genuinely trained LoRA adapter, with no hidden Fixed fallback credited to raw policy results. Nine physical actions provide history, three retrieval sources, filtering, ranking, verification, finish and clarification. Failed actions become observations; unchanged repeated actions are rejected. Finish requires enough verified candidates. The deadline is checked between actions, not an OS hard interrupt.

`DualRetriever` uses BGE item vectors and the frozen full-trained M3 Transformer. Generated Semantic IDs are catalog-valid and collision-expanded before item-level weighted reciprocal-rank fusion. Train-only Popularity is a recovery/cold-start source. Recall/NDCG and task-contract metrics are kept separate.

Executed teacher traces generate corrected v2 action supervision with disjoint users/price/goal groups. Only assistant targets are supervised. `assistant_action_loss` attends to full context but projects only logits that can contribute to the causal action loss; tests verify loss and gradient equivalence. The actual one-epoch Qwen LoRA run and weak held-out performance are preserved in [results_m5_sft_lab95.md](results_m5_sft_lab95.md).

## Delivery surface

`run_project.py check` checks artifacts, runtime and optional full cached-model hashes without downloads. `demo` invokes the existing real Agent in three preset scenarios and independently grades the products against catalog metadata. `report` reads existing metrics, rejects mismatched policy task IDs/source/model hashes or a mismatched adapter, and renders separate test-retrieval, validation-fusion and synthetic-policy sections. It does not rerun benchmarks or tune the models. All generated outputs must be in a fresh subdirectory of this project's `outputs/`; the separate RecAI project is not a dependency.
