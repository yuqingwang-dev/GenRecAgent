# Recovery SFT v3 plan — 2026-10-07

This is implementation/testing work, not a presentation milestone. The project goal remains unfinished. Actual v3 training has not started without the requested resource confirmation.

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
