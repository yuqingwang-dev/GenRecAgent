# Recovery SFT v3 plan and actual run — 2026-10-07/08

The plan below was frozen on 2026-10-07. The user explicitly approved this run on 2026-10-08. Training and the complete independent comparison have now finished, and their weights/logs/traces were returned and verified. Earlier waiting-for-confirmation statements below are historical, not the current training status.

## Frozen changes

- Raw compact observed-state protocol, maximum observed sequence length 446 tokens. No `phase`, `allowed_actions`, reference next action or controller answer is provided as input. Metadata counts and tool flags are observable facts.
- Train-only curriculum: 288 executed synthetic tasks plus physically checked partial recovery snippets. Select 1,080 actions (120 per tool); 1,035 distinct selected examples. Oversampling is disclosed, not counted as additional trajectories.
- Additional small-positive eligibility cases teach that a nonempty catalog can still be insufficient. Wrong-stage filtering, malformed retrieval parameters and premature finishes are actually rejected by the tool environment; only the physically accepted corrected action is supervised.
- Fresh validation/test: 72 tasks each, new users excluded from every v2 split, distinct goal price sets and distinct language templates. All task states still supply normalized intent/constraints: this is not a claim of broad free-form language understanding.
- Action-name tokens receive weight 4; other assistant tokens weight 1, context weight 0. Unweighted validation NLL remains diagnostic. Sixteen predetermined stratified **validation** rollouts select the checkpoint by feasible fill, total contracts, fewer illegal calls, then NLL. Test tasks are never read by the trainer.

## Bounded real-run proposal

Reuse lab95's pinned Qwen2.5-1.5B snapshot and existing real rank-8 q/v LoRA adapter. At most 1,200 train actions, two epochs, LR 0.0001, accumulation 8, BF16 base, max length 1,024, seed 42. No new foundation-model download, paid API or base-model update. Estimated 15–35 minutes for training/validation selection on the existing RTX 3090; independent full evaluations are additional inference. Previous optimized training peak was about 3.1 GiB; the launcher requires 5.5 GiB free and never interrupts another GPU process.

After approval, `run_lab95_recovery.sh` runs real training, full raw validation and frozen Fixed/Compact Prompt/Compact SFT on the fresh test split. No runtime controller is used for the compact comparison. Models and results are automatically returned and checksummed.

The launcher now checks every proposed training setting (including two epochs, rank-8 q/v LoRA, length 1,024 and sixteen validation selection tasks), actual action counts and input/adapter fingerprints before touching GPU execution. It refuses any existing training/job/evaluation/archive output rather than overwriting logs. The actual 1,080 actions imply at most 2,160 training action visits and 270 optimizer steps. `--check-only` exits before importing Torch or creating training outputs; this checks readiness, **not** permission.

On the already authorized lab95 project checkout, the following entry is read-only and does not start training:

```bash
bash scripts/run_lab95_recovery.sh /mnt/wangyuqing/genrec_upload/m3_server_upload_02efbf6/env/bin/python --check-only
```

## Evidence so far

Generated data: `data/processed/tool_sft_recovery_v3/manifest.json`. Local name-weighting/masking and data-isolation tests passed. On lab95, the random miniature-model CPU software test passed and the updated complete suite passed **120 tests**. An optional-runtime metadata bug in saved-index evaluation was fixed rather than installing an encoder unnecessarily. The full frozen dual-retrieval test completed on lab95 after a local Windows native-process crash; its result is kept separate from model-policy tasks.

Compact Prompt validation is a same-protocol baseline, not SFT. It does not use test tasks or a controller. The old v2 results and model hashes remain available and are not relabeled as v3.

The baseline finished at 0/72 contracts, with no feasible fills or correct clarifications. Its records, the 120-test pass log and the miniature CPU software check were automatically returned to `outputs/recovery_v3_pretraining_server`. This is not a trained v3 adapter. The requested lab95 real-training confirmation remains pending.

Subsequent acceptance work added full candidate artifacts, explicit end reasons/rate denominators, physical source-removal experiments and an offline evidence validator. The latest full lab95 suite passes 138 tests; three sequential Fixed runs on this same fresh validation split pass 72/72 each with complete pool provenance. These system checks do not replace raw v3 SFT or its new held-out comparison; see [the acceptance audit](acceptance_audit_2026_10_07.md).

Later preflight hardening found and fixed an eager Semantic-ID package import: plain data storage had unnecessarily imported Torch on lab95. Public APIs now lazy-load their optional runtimes. Fresh subprocess tests prove ordinary storage/codebook/preflight imports do not load Torch, NumPy or Transformers; the ML tests still exercise their real implementations. The updated complete lab95 suite passed **154 tests in 6.45 seconds**, exit 0. Its real read-only launcher and invalid-approval rejection also passed, with no test content opened and no training output created. Returned evidence is `outputs/recovery_preflight_server/recovery_preflight_checks_final`, SHA256 archive `3e940dd1d3e8989aeaa6a47863a71e8aa36b8f9038187ca91da07a3ce35004f8`. This does not fix or certify the separate Windows native Torch instability during real inference.

Full frozen test retrieval completed for 8,151 users, with zero source failures: Dense Recall@20 0.036560; Transformer 0.002454; 50% Dense fusion 0.031407; 75% Dense fusion 0.036928. Source endpoints exactly reproduce the original test runs. The 75% setting adds only three net hits; no robust statistical gain is claimed and weights were not tuned on test. `outputs/dual_retrieval_frozen_test` contains all Recall/NDCG@5/10/20 and provenance.

## Actual approved run — 2026-10-08

The single lab95 job exited 0. Training used the unchanged proposal: 1,080 selected actions (1,035 distinct), two epochs, rank-8 q/v LoRA and action-name weight 4. Recorded training/validation-selection time was **694.76 seconds** (excluding initial model/data loading); CUDA peak allocated memory was 3,240,066,560 bytes, about 3.02 GiB. The predeclared sixteen validation rollouts selected **epoch 1**: 16/16 contracts, eight feasible fills and no illegal decisions. Unweighted validation NLL was 0.724097 before training and 0.004506 for the selected checkpoint. Both epochs executed; the second epoch was not automatically assumed better.

| Fresh full test (72 identical tasks) | Contracts | Feasible filled | Correct no-result | Illegal decisions | Mean steps | Mean loaded execution ms | Output tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed | 72/72 | 36/36 | 36/36 | 0 | 6.972 | 124.65 | 0 |
| Raw Compact Prompt | 0/72 | 0/36 | 0/36 | 682 | 10.000 | 20,062.47 | 41,076 |
| Raw Compact SFT | 72/72 | 36/36 | 36/36 | 1 | 4.542 | 3,096.91 | 5,107 |

All policies used the same catalog, recommendation checkpoint, original tasks, ten-action/120-second-between-actions limits and sequential RTX 3090 execution. Prompt and SFT share the same compact input, BF16 and 1,024/96 token limits. SFT used **no runtime controller, action mask or fixed-flow fallback**. Its sole illegal decision was rejected and recovered by another model decision. Backend failures and externally seeded premature finishes remain separate from model illegality. Legal decision rate was 314/315 = 99.68%; mean actual tool attempts was 4.375. All 36 nonempty SFT outputs satisfied the declared constraints and complete candidate provenance checks. Prompt returned no items: its conditional nonempty constraint rate is **N/A**, not 100%.

Full post-training validation was 70/72 (34/36 feasible fills and 36 correct no-result outcomes). The two warm hybrid failures had only two eligible retrieved items, performed rank/verify before Popularity recovery and exhausted the unchanged ten-step budget. Their traces are preserved. This loss/rollout mismatch and imperfect validation prevent a universal-success claim despite this test grid's 72/72.

The observer supplies normalized route/constraints, tool flags/counts and a static full-catalog eligibility count. Learning when to clarify from that supplied fact is not evidence of autonomous discovery of unknown catalog feasibility. Tasks are synthetic, user/template/price-disjoint and single-seed; they are not real online requests or broad free-form planning. The raw Prompt baseline is deliberately untuned under the shared custom JSON schema, not a strong optimized prompted-agent benchmark. This comparison does not isolate the individual contributions of compact inputs, curriculum and name weighting, and cannot be merged with v2's different task protocol. Fixed remains much faster. RQ-VAE collapse and weak learned recommendation quality are unchanged.

Training/evaluation code commit: `b452480ce719940c5e61e9e04e5b99285f9f4c21`. Selected adapter SHA256: `68746429318ede24c5150f2a26698b097a06f0ec7b801d4dd3ce6bafcb77de26`. Returned archive SHA256: `9cace2772c4394b684774464939c9335c56f34c44a47ea0e473ae423a909a125`. The old adapter is preserved. The independent CPU report verifies base/input/source fingerprints, checkpoint score, **112/112 changed finite tensors**, 1,089,536 saved tensor elements and all **288 full validation/test executions** against original tasks and metadata. Canonical artifacts are `outputs/tool_sft_recovery_v3_server` and `outputs/recovery_v3_delivery_report_verified`.

After the benchmark was frozen, a malformed-name handler edge case and caller-mutable action trace were hardened. These changes do not retroactively replace benchmark code: source verification checks the byte-identical preserved recorded Git commit when current files differ. New unified `compact_sft` demo defaults to the returned checkpoint; legacy `sft` remains v2. No additional training was started.

Final software at `92ad498` passed **173 tests in 6.72 seconds** on lab95. The actual raw-Compact-SFT unified CLI also passed all three preset contracts: hybrid history recommendation (five products, seven steps), cold-start Popularity (five products, six steps), and infeasible budget (clarification, one step), with no controller. These are real demonstrations, not another benchmark. Their checksummed logs and traces are local in `outputs/recovery_v3_cli_server/recovery_delivery_checks`; archive SHA256 is `5fc8e62479a1a098e9ca52c697abb409cf8f8dac3efbf662fad40b6479d3ce75`. The current complete CPU evidence report is `outputs/recovery_v3_delivery_report_final/results.md`. Local Ruff and the new runtime-independent tests also pass. Windows native GPU execution remains separately unguaranteed; the three demonstrations were actually exercised on lab95.
