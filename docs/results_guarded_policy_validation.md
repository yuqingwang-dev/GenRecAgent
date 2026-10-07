# Guarded model runtime — validation checkpoint (2026-10-07)

This is a runtime change, not further LoRA training. It reuses the existing pinned Qwen base and returned adapter. The old 12-task test result remains raw SFT **4/12**; no test result was overwritten or used to choose the guard's rejection budget.

## Implemented behavior

- Compute permitted next actions from observable history, attempted sources, candidate counts, filtering/ranking and verification flags. Do not read expected task labels.
- Reject malformed arguments, retrieval counts below the requested count, wrong-stage tools and premature finish/clarification before physical execution.
- If catalog metadata cannot supply the requested count or the normalized request is rejected, use an explicit environment/controller clarification rule without model generation.
- Substitute a logged controller action for each rejected proposal; two cumulative rejections trigger controller handoff for the rest of the request. The threshold was fixed before this run.
- Preserve each model proposal, rejection reason, executed action source and generation record. Physical tool failures remain actual observations, not model rejection counts. No constraint is weakened.
- Preserve ten physical actions and the between-actions 120-second deadline; it is not an OS interrupt. A separate regression test verifies generation cost survives an expiry before physical execution.

Raw modes and `ToolEnvironment` are unchanged. `guarded_sft`/`guarded_prompt` are separate optional policies; Fixed remains the default. The single-request and unified entry points are both connected.

## Actual paired validation

All 72 existing validation tasks, with 36 feasible and 36 infeasible requests, were evaluated sequentially on the same local RTX 4060 Laptop runtime. Task IDs, task-file hash, source hashes, recommendation checkpoint and SFT adapter hashes match across the comparison. The model is the same one-epoch adapter; no weight update, prompt adjustment or threshold sweep occurred. Scope is normalized synthetic tasks with reused language templates, not broad free-form intent interpretation.

| Mode | Contracts | Feasible filled | Correct clarification | Executed policy-illegal calls | Tokens | Mean request ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed | 72/72 | 36/36 | 36/36 | 0 | 0 | 112.76 |
| Raw SFT | 26/72 | 4/36 | 22/36 | 337 | 7,946 | 11,061.48 |
| Guarded SFT system | 72/72 | 36/36 | 36/36 | 0 | 2,946 | 4,188.02 |

All modes had zero measured returned-item price/category/history violations. Raw SFT returned only 20 items; the other modes returned 180. Empty output is not counted as feasible fulfillment. The guarded run has 18 failed physical calls, including seeded early finishes/source faults, which are distinct from executed policy-illegal calls.

Guarded contribution accounting:

- 204 model proposals: 150 accepted for execution, 54 blocked.
- 166 controller actions; **72/72 tasks involved the controller**.
- 18 tasks crossed the two-rejection handoff threshold.
- **0 unassisted guarded contract passes**. The 72/72 system result must not be attributed to raw SFT or autonomous learned planning.

Mean latency excludes model loading and independent grading; it includes environment setup, model generation and tools. It is one machine-specific run, not a throughput confidence interval. Guarded SFT is still much slower than Fixed for the same contracts.

Evidence: `outputs/guarded_policy_validation_v1/{fixed,raw_sft,guarded_sft}/{metrics.json,traces.jsonl}`. Run source commit `8767328`; the subsequent timeout-cost patch affects only the tested expiry boundary, which did not occur in these completed runs. Existing raw and guarded generation records remain sufficient to reproduce all token totals. The adapter SHA256 is still `ab9f93228affe88b7e281b14ad8a1cfd88fba299226d80d9c9ac5ac1dce16f56`.

`outputs/guarded_policy_runtime_v1` additionally exercises the unified guarded entry on real-catalog hybrid, cold and infeasible-budget requests. These are integration checks, not another held-out benchmark.

Final software regression: 112 tests passed and Ruff passed. One initial Windows process crashed while importing the native regex dependency during test collection; a fresh process with explicit Torch/regex/Transformers preloading passed the complete suite. No CUDA/runtime package was reinstalled or changed. A one-task validation smoke also exercised the final token-accounting CLI patch.

The next model-quality issue remains state/precondition and recovery learning. Validation outputs are diagnostic records, not a new training corpus; additional training should use train-only tasks and remain separate from held-out evaluation, on lab95 as requested.
