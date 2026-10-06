# Minimal completion checkpoint — 2026-10-06

The goal of this increment is a small, executable research pipeline using the existing catalog and models. It does not add a frontend, another foundation model, RL or elaborate re-ranking. Real SFT is the remaining acceptance gate, not an already completed experiment.

## Actual dual retrieval

`DualRetriever` uses the saved BGE index and the **trained M3 Transformer**, expands Semantic ID collisions into catalog items and fuses item ranks with weighted reciprocal-rank fusion. Raw cosine scores and token log probabilities are not directly added. Sources are deduplicated and seen/ineligible items are excluded.

All 8,151 validation users, full 112,590-item catalog, 50 candidates per source, existing 20-code constrained beam, rank offset 20; weights were declared before the run. This table is **validation only**, not a new test result.

| Method | Recall@20 | NDCG@20 |
| --- | ---: | ---: |
| Dense | 0.046865 | 0.027733 |
| Trained Transformer | 0.003312 | 0.001193 |
| 50% Dense / 50% generated | 0.040486 | 0.022894 |
| 75% Dense / 25% generated | 0.046865 | 0.027803 |

No source failures were observed. Dense and generated top-50 lists overlap by only 0.305 items on average. The 75% fusion changes ordering slightly without adding Recall@20 hits; the tiny NDCG difference is not a demonstrated robust improvement. Dense remains the default personalized route. Hybrid is an explicit experimental option.

Evidence: `outputs/fast_dual_validation/{metrics,run_config}.json`, code commit `9e809d6`, frozen Transformer SHA256 `dba260171051311a4e214947bfdaec2449e8cabad0ea6e35907076fb08422dd3`. M1 and M3's original test results remain in their separate reports.

## Real multi-step environment

The nine actions are `read_history`, `retrieve_dense`, `retrieve_generative`, `retrieve_popular`, `filter_candidates`, `rank_candidates`, `verify_candidates`, `finish` and `clarify`. Retrieval actually executes against the catalog. Candidates are checked for provenance, membership, history exclusion, duplicates, finite scores, sequential ranks, static price/category eligibility and required count. `finish` cannot bypass verification. Unknown prices are excluded under a price budget.

Errors and new observations support subsequent actions; unchanged repeated calls are rejected. Physical source failure may recover with another source or train-only Popularity. Constraints cannot be weakened by model arguments. Unsupported normalized requests can only clarify. Maximum 10 actions, with a 120-second deadline checked between actions, **not an interruptible OS timeout**.

`FixedStepPolicy` is deterministic state-aware orchestration and the trajectory teacher. `QwenStepPolicy` is actual local model generation of the next JSON action. There is no hidden rule fallback counted as raw model success. Natural-language intent compilation is separate from this action-policy comparison.

## Paired task comparison and ablations

Use the SHA256-sorted first 12 of 72 held-out tasks, chosen independently of outcomes. Every method uses the same IDs and test-file hash `b553a4b73d1ab1b8d26b9c3f7d6a755c83bea0345ec789984bd9a7786815bc38`. The subset has nine feasible tasks and three catalog-infeasible tasks. Tasks cover cold/known history, price, the exact `Premium Beauty` catalog tag, injected Dense failures and deliberately premature termination.

The grader independently reads item metadata and requires five valid products for success or correct clarification when fewer than five eligible catalog products exist. Returning no items is not counted as constraint-satisfying task success. The no-verification ablation removes only the finish gate; the grader remains enabled. No-recovery stops after a failed tool action. Deliberately injected failures are separated from policy-illegal calls.

| Policy / ablation | Passed contracts | Policy-illegal calls | Generation tokens | Mean execution ms |
| --- | ---: | ---: | ---: | ---: |
| Fixed + checked finish + recovery | 12/12 | 0 | 0 | 70.67 |
| Raw prompted Qwen, same environment | 0/12 | 101 | 3,696 | 11,259.84 |
| Fixed, no recovery | 7/12 | 0 | 0 | 69.13 |
| Fixed, no finish verification | 10/12 | 0 | 0 | 75.68 |
| Real LoRA SFT | not run | — | — | — |

Fixed successfully fills all nine feasible tasks and correctly clarifies the three infeasible ones. Raw Qwen exhausts the ten-action budget on every task, returns zero products and has 106 failed calls including the five externally seeded faults; zero returned-price violations is therefore **not** success. No-recovery makes the three source faults and two early wrong finishes terminal. Removing finish verification allows two empty early finishes to falsely claim success. Execution latency includes environment initialization/action inference and real tools, excludes model loading and independent grading, and is specific to this local RTX 4060 run.

Final immutable-observation traces and machine-readable metrics are in `outputs/fast_completion/{fixed,prompt,no_recovery,no_finish_verification}`. Code commit `79bd246` and per-file hashes are recorded in each run. Full generation text, token counts, illegal-call counts, latency, task IDs and model revision are retained. The 12-task table is a small synthetic acceptance subset, not a broad benchmark, user study or recommendation-relevance score. No test-driven prompt adjustment is made.

The CLI example in `outputs/fast_completion/demo.json` executes both real sources and the filter/rank/verify/finish loop with a $20 static-price limit. It is a trace example, not another recommendation-quality experiment.

## Executed trajectory data and SFT readiness

Dataset `data/processed/tool_sft_v2` was generated by running and independently checking the real tools, not fabricating candidate observations.

| Split | Tasks | Supervised assistant actions |
| --- | ---: | ---: |
| Train | 96 | 608 |
| Validation | 72 | 492 |
| Test | 72 | 490 |

Users and complete mode/price/category goal combinations are disjoint across splits. Price values are also disjoint. The same synthetic language template is reused: **this is not template-held-out free-form language evaluation**. Input states already contain normalized intent, budget and category; SFT trains next-action selection/arguments/termination, not the language parser or recommendation ranking model. It is teacher imitation, not online user feedback.

Only assistant action and its end marker receive labels. User, instruction and observation prefixes receive `-100`; overlength samples are rejected, not silently truncated. Maximum observed length: 919 tokens under the pinned Qwen tokenizer. Wrong injected early `finish` actions are kept as observations but are not supervised targets; valid recovery actions are supervised.

The old v1 export had a mutable error-dictionary snapshot problem: later errors could alter earlier saved states. Snapshots now copy that data and nested last observations; v2 was regenerated and checked for future-fault leakage. **The training entry refuses v1**. No real SFT used the old data.

`scripts/train_tool_sft.py` reads only train and validation action files, checks their hashes and split metadata, loads the already cached Qwen revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` and trains only q/v LoRA adapters. `configs/tool_sft.yaml`: rank 8, alpha 16, one epoch, gradient accumulation 8, learning rate 0.0002, seed 42, max length 1,024, CUDA BF16. Validation action loss selects the saved adapter; the test task suite is evaluated separately. Output: `adapter/`, `metrics.json`, `training_log.json`, `run_config.json`.

The CPU software check in `outputs/fast_completion/cpu_software_smoke` used a **random one-layer miniature model**, two train samples and one reused validation sample. It validates loss/backpropagation/masking/adapter serialization, not generalization or pretrained-model SFT. Both CLI and evaluation reject a `software_smoke_only` adapter as a full SFT result.

Resource proposal awaiting approval: reuse the local 1.5B snapshot, train 608 actions for one epoch on the local RTX 4060 Laptop 8 GB, save only the adapter inside this project. Estimated several minutes to half an hour and roughly 5–7 GB VRAM; these are estimates, not measured full-run costs. The laboratory server remains an alternative. There is no paid API use or new foundation-model download. The earlier explicit user instruction requires confirmation before this real training step.

## Finish the minimum version

After approval, run one real SFT and evaluate its adapter on the same declared 12 tasks with unchanged prompt/tools/grader. Report losses, contract pass, illegal calls, tokens and latency alongside Prompt and Fixed, even if SFT does not win. Archive the model/data/config hashes and add that measured row here. More training, larger independent language sets, better RQ-VAE/generative ranking, confidence intervals, a frontend and public artifact packaging are subsequent refinements.

Software regression at this checkpoint: **93 tests passed**, whole-project Ruff lint passed, and newly changed Python files passed formatting checks. Legacy files were not reformatted. No changes to the separate RecAI project are required.
