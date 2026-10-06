# Real LoRA SFT and paired evaluation — 2026-10-06

**Actual training and artifact return are complete.** The adapter is genuinely trained from the cached Qwen base, not the random CPU software-smoke model. The minimal experiment chain is now runnable end to end; learned policy reliability is still inadequate, so Fixed remains the default demonstration policy.

## Approved run and provenance

The user explicitly requested lab95 for training. A new isolated checkout was created at `/mnt/wangyuqing/genrec_upload/sft_lab95_7048bf0/GenRecAgent`. The historical M3 checkout/results were retained. Its project-specific Python environment was reused, keeping Torch `2.7.1+cu118` unchanged and adding Transformers `5.17.0`, PEFT `0.21.2`, Accelerate `1.15.0`; `pip check` passed. GPU: RTX 3090. No other GPU process was terminated.

Training and all five policy/ablation evaluations used source commit `0adadbd187f2c47859e5308d035162989a1535c7`. Code/source hashes, seeds, dataset manifest, versions and GPU are saved in `tool_sft_lora/run_config.json`. The detached runner exited **0**, and its logs were returned.

Base: Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; weights SHA256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`. The existing local snapshot was uploaded; every snapshot-file hash was checked before training. The server Dense index was reused only after matching its saved metadata and embeddings with the local copy. The frozen M3 Transformer and catalog/data artifacts were transferred with the same hashes as the existing experiments.

## Data and actual training

Corrected `tool_sft_v2`: 96 train tasks / 608 actions, 72 validation tasks / 492 actions, 72 test tasks / 490 actions. Users, price values and complete mode/price/category goal combinations are disjoint. The trainer reads **train and validation action files only**, not test actions. Action-file hashes and immutable-observation flag are checked. The old mutable-observation v1 data are refused.

Configuration: q/v LoRA rank 8, alpha 16, one epoch, learning rate 0.0002, gradient accumulation 8, max sequence 1,024, seed 42, BF16 base, frozen base parameters. There are **1,089,536 trainable adapter parameters**. Only assistant action/end-marker labels are trained; instruction/user/observation tokens remain masked. Full context is still attended to, while `logits_to_keep` avoids projecting masked prompt positions. CPU tests establish equivalent causal loss **and gradients**, including the first action token's shift.

| Measured training quantity | Value |
| --- | ---: |
| Initial validation action NLL | 0.640970 |
| Epoch-average train action NLL | 0.112467 |
| Final validation action NLL | 0.038242 |
| Train/validation passes + adapter-save time | 401.46 seconds |
| Peak CUDA allocated memory | 3,335,912,448 bytes (3.11 GiB) |
| Peak CUDA reserved memory | 3,546,284,032 bytes (3.30 GiB) |

Time excludes package preparation/model transfer and initial base loading. Lower action NLL is not itself a task-success or recommendation-quality result. Train NLL averages losses while the model is changing; final validation NLL is measured after the epoch. There was no further training, prompt adjustment or hyperparameter search based on the held-out outcomes.

Adapter SHA256: `ab9f93228affe88b7e281b14ad8a1cfd88fba299226d80d9c9ac5ac1dce16f56`.

## Same-machine task comparison

Every row uses the same SHA256-selected first 12 of 72 held-out tasks, same tools/prompts/grader, frozen recommendation checkpoint and test-file hash `b553a4b73d1ab1b8d26b9c3f7d6a755c83bea0345ec789984bd9a7786815bc38`. There are nine feasible requests and three catalog-infeasible requests. Model policies use deterministic local FP16 inference on the same RTX 3090. This is a separate same-machine comparison; do not mix its timings with the earlier local RTX 4060 table.

| Policy / ablation | Contracts passed | Feasible filled | Correct clarification | Policy-illegal calls | Generation tokens | Mean execution ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed | 12/12 | 9/9 | 3/3 | 0 | 0 | 171.35 |
| Raw Prompt | 0/12 | 0/9 | 0/3 | 101 | 3,704 | 12,264.39 |
| Raw SFT | 4/12 | 1/9 | 3/3 | 46 | 1,276 | 5,620.05 |
| Fixed, no recovery | 7/12 | 6/9 | 1/3 | 0 | 0 | 129.55 |
| Fixed, no finish verification | 10/12 | 9/9 | 1/3 | 0 | 0 | 153.28 |

Synthetic seeded tool failures/early wrong termination are excluded from the policy-illegal count. The evaluator independently checks actual catalog metadata, history exclusion and five returned products; it does not trust the Agent's success label. SFT has seven clarifications (four on feasible tasks), four exhausted budgets and one genuine successful recommendation. Its 50 failed calls comprise 38 unchanged repeated actions, eight required-source precondition failures, two externally seeded wrong finishes and two observed injected Dense failures. Another injected Dense-failure task never reaches retrieval. This is a stateful action-selection failure, not evidence of a broken adapter file.

All methods have zero measured returned-item constraint violations, but Prompt returns **zero items**, SFT **five**, and Fixed **45**. Empty output is not successful constraint satisfaction. The finish-gate ablation falsely claims success for two empty infeasible requests; the unchanged independent grader rejects them. No-recovery fails on the externally seeded fault cases.

Latency includes task/environment initialization, generation and actual tools, excludes model loading and independent grading. It is one sequential run on a **shared GPU**, not a controlled throughput benchmark or confidence interval. Raw SFT is not promoted to the default policy.

## Automatic return and local loading

Result archive SHA256 `5c7c4d0865732d25382df99d2a3f9e6f331e960d8b2c2d45833134c3d679f6ed` was compared before local extraction. Adapter SHA256 was rechecked after return. **No manual XFTP transfer is needed.** Local artifacts:

- `outputs/tool_sft_lab95_server/tool_sft_lora/adapter/`: trained adapter, adapter configuration and tokenizer.
- `outputs/tool_sft_lab95_server/tool_sft_lora/{metrics,run_config,training_log}.json`: actual training evidence.
- `outputs/tool_sft_lab95_server/lab95_comparison/{fixed,prompt,sft,no_recovery,no_finish_verification}/`: paired metrics and complete traces.
- `outputs/tool_sft_lab95_server/lab95_job/`: driver, training and evaluation logs, exit code.
- `outputs/sft_lab95_upload_7a5b881/transfer_receipt.json`: transfer checksums and completion receipt; no password is saved.

The returned adapter loaded successfully with the existing local base model. A **software/demo check**, not another held-out benchmark, ran hybrid retrieval under a $20 static-price limit: `local_adapter_demo.json` returned five verified products in nine steps with two precondition errors recovered by the model. `local_fixed_demo.json` returned five products in seven steps without invalid calls. These do not override the 4/12 held-out result.

```powershell
conda activate genrecagent
$env:PYTHONNOUSERSITE = "1"
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:HF_HUB_OFFLINE = "1"
python scripts/run_tool_agent.py --policy fixed --mode hybrid --history-item B01DNS649G --max-price 20 --output outputs/new_fixed_demo.json
python scripts/run_tool_agent.py --policy sft --adapter outputs/tool_sft_lab95_server/tool_sft_lora/adapter --mode hybrid --history-item B01DNS649G --max-price 20 --output outputs/new_sft_demo.json
```

These commands perform inference only, not local training. Use fresh output paths. Heavy data, base weights and adapters remain ignored by Git; a public clone alone does not contain the cached artifacts.

## Minimum delivery and remaining limits

Implemented/measured: data and retrieval baselines, usable explicitly named K-Means Semantic IDs, real Transformer generation with constrained decoding/collision expansion, actual item-level dual retrieval, stateful catalog tools, fixed/raw-model action policies, verified trajectories, **real LoRA SFT**, same-task policy/verification/recovery comparisons, artifact provenance and CLI demos. Full local regression: **95 passed**, Ruff passed.

For rapid completion, use the fixed pipeline for reliable demonstrations and keep SFT as a measured experimental branch. No frontend, RL, additional foundation-model download, test-driven retraining or claim of a repaired RQ-VAE is added. Remaining quality work includes better state/recovery generalization, independent natural-language templates and larger evaluations, stronger generative relevance and release packaging. Task states already supply normalized intent/budget/category and reuse synthetic templates; neither 4/12 nor Fixed 12/12 proves broad language understanding, online impact, production readiness or a new recommendation algorithm.
