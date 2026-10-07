# Requirement-by-requirement acceptance audit — 2026-10-07

Scope is the original GenRecAgent request, not a smaller demonstration project. The latest user also requires continued development/testing. The preceding goal turn made concrete progress (new data, software validation, complete frozen fusion test); this continuation audited the remaining requirements and filled metric/audit/source-ablation gaps. The goal is **not complete** and additional real recovery SFT still needs the explicitly requested training confirmation.

## Classification

A: implemented and actually exercised. B: implemented but intended verification is incomplete. C: mock/placeholder not used by the real path. D: missing. An A training run is not proof that its model quality goal succeeded.

| Original requirement | Direct authoritative evidence inspected | Finding |
| --- | --- | --- |
| Independent repository/environment, preserve existing work | GenRecAgent Git root/main; clean start; dedicated genrecagent interpreter and lab95 project environment | A; RecAI source not accessed or modified |
| `parent_asin`, missing metadata/price, duplicates/times, chronological history | `data/amazon.py`, `storage.py`, M1 prepared manifest, data tests | A; reviews are not purchases; static catalog/global-time limitations remain documented |
| Popularity/Dense, full catalog, all Recall/NDCG K | `outputs/m1_all_beauty_full/{metrics,run_config}.json`, saved index; NumPy exact search tests | A; FAISS-specific backend remains B, but allowed exact-equivalent NumPy is measured |
| RQ-VAE training/mapping/reconstruction/usage/collisions | `outputs/m2_rqvae_full`: actual `rqvae.pt`, item map, metrics and training log | A implementation/run, **quality failed** (collapse); no repaired-RQ-VAE claim |
| Simplified tokenizer separately named | `outputs/m2_rqkmeans_full` codebooks/projection/item map/run metrics | A independent baseline; not substituted as RQ-VAE success |
| Real historical Transformer/training masks/validation selection | M3 train-only prefixes, `transformer.pt`, run config, best epoch 5, 10,969 train examples; mask tests | A; recommendation result is weak, not an algorithm win |
| Catalog-valid code generation/collision expansion/seen exclusion | retrieval trie/generative/transformer code and candidate tests; full test diagnostics | A; target-independent expansion and train popularity tie handling |
| Real Dense/Transformer fusion, no raw score addition | `retrieval/hybrid.py`, validation and frozen 8,151-user test metrics | A; RRF, same source endpoints; 75% test difference only 3 net hits, no robust-gain claim |
| Qwen actual schema calls and three separate modes | `LocalQwenLLM`, real Fixed/Prompt/SFT test/validation traces, pinned base and adapter hashes | A; raw SFT reliability weak; guarded controller result is separately attributed |
| Actual history/retrieval/filter/rank/verify/finish/clarify | nine-action `ToolEnvironment`, real-catalog runtime runs | A; model cannot execute Python/Shell/SQL |
| Candidate membership/source, price/category/history/duplicates/count/rank | production verifier; new independent `agent_contracts.py`; full candidate artifact audits | A implementation; new runtime artifacts now permit offline pool membership checks; old preview-only traces cannot retroactively prove full pool membership |
| No output/unsupported conditions never loosen constraints | controlled M4 tasks, metadata grader, raw/guarded runtime tests | A for declared USD/static-tag scope, not broad unconstrained language or live prices |
| Budgets, repetition, recoverable failures/timeouts, explicit end reason | ten-action loop, between-action deadline, repetition tests, actual `TimeoutError` recovery fixture, end-reason helper | A for this bounded local protocol; hard OS interruption of an in-flight GPU call is **not implemented** |
| Verified, sourced trajectories with varied outcomes | v2 and v3 actual execution manifests/traces; train-only wrong-stage/repeated-error snippets | A; synthetic tasks and partial repairs explicitly labeled |
| Isolation of users/templates/constraints and action-only labels | v3 manifest, 1,080 selected/1,035 distinct rows, new held-out users/prices/templates, mask/name-span tests | A new data/protocol; old v2 reused templates remain a disclosed limit |
| Real SFT with logs/config/checkpoint and raw selection | v2 real lab95 adapter/metrics; v3 weighted trainer + raw validation selector + random CPU software check | A v2 run; **B v3 real training**, not started, cannot be declared completed |
| Two distinct experiments and fair policy comparisons | full recommendation A; normalized task B with same devices/data/model/budgets and independent grader | A existing comparisons; new v3 trained-model/fresh-test comparison is B pending real training |
| Verification/replanning/source-removal ablations | existing finish/recovery results; new physical-source removal cases on same v3 validation requests | A; quantity contracts do not measure relevance |
| Explicit rates/calls/steps/latency/tokens with proper denominators | original metrics plus `agent_contracts.py`, `supplement_agent_metrics.py`, supplement files | A definitions/tests; empty-output constraint rate is N/A, not 100%; service failures are separate from illegal model calls |
| Success/failure examples/CLI demo/limits/evidence ledger | real traces, unified CLI, results reports, resume ledger and interview notes | A; no public release or fabricated online lift |
| No heavy artifacts/secrets in Git, no push | ignore rules, share audit, clean Git snapshots and absence of publish action | A current snapshot; full historical secret/license certification is not claimed |
| Own vs external implementation/license/source | own MIT code; cached Qwen license; dataset/model source links; explicit BGE/Qwen/PEFT/sklearn attribution | A documented attribution; external artifacts are not redistributed as owned algorithms |

## Concrete remaining gates

1. Obtain confirmation for the bounded two-epoch lab95 recovery run (existing 1.5B base/rank-8 adapter, 1,080 selected actions; no download/paid API). Do not treat an automatic goal continuation as approval.
2. Execute real v3 training and checkpoint selection from the predeclared 16 validation rollouts; verify adapter hashes, finite losses, frozen base and input protocol.
3. Freeze the selected adapter before new held-out test evaluation. Compare Fixed, same-input Compact Prompt and raw Compact SFT, with no hidden controller. Automatically return results and preserve failures if SFT does not improve.
4. Re-run this completion audit against the final artifacts. RQ-VAE collapse, weak learned recommendation quality, static catalog/global-time limits and soft rather than interruptible timeouts must remain explicit. Do not declare all quality objectives satisfied from software tests or a controller's 100% score.

This continuation is not a new model-training result. Additional safe software/evidence work has been performed while waiting for confirmation; no newly trained checkpoint exists yet.

## Verified acceptance outputs

The complete lab95 suite passed **126 tests in 17.94 seconds** at `fce87a5`; adding the offline evidence auditor then passed **138 tests in 17.09 seconds** at `9bc0be2`, exit code 0. Locally, its twelve runtime-independent tests passed in 0.06 seconds and Ruff passed. The local earlier native Torch process crash is not presented as a clean complete test run.

Three **sequential** Fixed runs use the same frozen v3 validation task file (SHA256 `73af17257aed4cc82a49d34df36b66c4ad06d3a85e4a77ca53dfb6eb2d1a089c`), recommendation checkpoint, catalog, budget and route requests. Disabling a source records its actual failure and allows normal remaining-source/Popularity recovery; it does not silently replace the requested route.

| Available sources | Contracts | Nonempty results / returned items | Mean agent tool attempts | Successful tool calls | Complete pool audits |
| --- | --- | --- | --- | --- | --- |
| Both | 72/72 | 36 / 180 | 6.806 | 480 | 228 |
| Dense disabled | 72/72 | 36 / 180 | 6.722 | 444 | 192 |
| Generative disabled | 72/72 | 36 / 180 | 6.528 | 430 | 178 |

Each run fills 36/36 feasible goals and correctly clarifies 36/36 no-result goals. All nonempty outputs satisfy declared static price/category/history/count constraints. Legal action attempt rate is 1.0: source backend failures are legal attempted calls, not illegal parameters. The externally seeded premature finish is excluded from policy-action denominators but remains visible in failed-call/step traces. None of these results measures recommendation relevance or autonomous Qwen improvement.

The portable results package was automatically returned to `outputs/acceptance_audit_server/portable/acceptance_audit`; its archive SHA256 is `79d49139e41543506206bb98ddd4d2e8c599e6db1d6fe82813cd7ec5bbbdd367`. The original archive also included Linux pytest symlinks incompatible with Windows extraction; it and its partial fixture extraction are preserved, but are **not** the canonical evaluation input. The evidence-only replacement contains ordinary logs/JSON files and passed complete checksum/path/type checks and extraction.

The subsequent 138-test log and the independently executed lab95 evidence check were returned to `outputs/acceptance_audit_server/final/acceptance_audit`, with archive SHA256 `479e478b6e85080884e5920d96f1881f479abe1e527f75e5c0ba25a825b2366a`. Both server and local checks pass and agree on all task/metrics/trace fingerprints and candidate/provenance counts. Local final JSON is `outputs/acceptance_audit_server/evidence_check_final/evidence_check.json`. No v3 trained-adapter files were present in the staged lab95 checkout when it was checked after these jobs.

`scripts/check_agent_evidence.py` independently regrades every execution against the original task file and local catalog. It checks recorded task/source/checkpoint fingerprints, complete retrieval-to-filter candidate chains, preview consistency, nonempty final-pool membership, end reasons, counts and conditional rate denominators. On all three returned runs it passed: **216 executions of the same 72 tasks**, 598 complete pool audits and 108 proven nonempty pool memberships. The twelve tests reject changed task labels, fabricated filter pools, mismatched previews, false rates/grades/end reasons, omitted tasks, wrong output counts, changed comparison budgets and unearned zero-decision rates.

Reproduce this **offline evidence consistency check**, with no training/inference or old-output overwrite:

```powershell
python scripts/check_agent_evidence.py --input-dir outputs/acceptance_audit_server/portable/acceptance_audit/paired_none --input-dir outputs/acceptance_audit_server/portable/acceptance_audit/paired_no_dense --input-dir outputs/acceptance_audit_server/portable/acceptance_audit/paired_no_generative --task-file data/processed/tool_sft_recovery_v3/validation_tasks.jsonl --output-dir outputs/new_acceptance_evidence_check
```

The checker verifies consistency of recorded artifacts, not a new retrieval execution, statistical quality proof or cryptographic proof of their origin. It deliberately rejects legacy preview-only traces instead of inventing complete provenance for them. Those older runs retain separately labeled supplements. The missing real v3 training/fresh-test result remains the next material acceptance gate.

## Final preflight before the authorization gate

The next continuation audited the actual launch path rather than treating green tests as completion. Its previous guard only capped the declared action count; it did not enforce epochs or prevent job-log overwrite. `recovery_preflight.py` and `check_recovery_run.py` now pin the proposed model/config/protocol, count the actual checked actions, verify train/validation files and initial adapter, and require entirely new training/evaluation/archive targets. No held-out test content is opened by preflight. The launcher has an exercised `--check-only` branch and rejects a missing/invalid approval flag before the run.

That real server check caught an unintended eager import from data storage through the Semantic-ID package to Torch. Lazy public exports preserve the existing APIs while avoiding GPU/NumPy/Transformers runtime imports for ordinary data/codebook checks. Local lightweight tests passed **28/28** and Ruff passed. Full lab95 ML regression passed **154/154 in 6.45 seconds** at `b452480`, exit code 0; the real preflight also confirmed `torch_imported=false`, config/manifest/initial-adapter fingerprints, no new training output and rejection of `--unapproved` with code 64. An earlier failed preflight/assertion log is preserved rather than overwritten. Local initial pytest temporary-directory setup errors were rerun using fresh project-contained directories; they are not misreported as passing runs.

The successful server evidence package was returned and SHA256 checked: `3e940dd1d3e8989aeaa6a47863a71e8aa36b8f9038187ca91da07a3ce35004f8`. Canonical files are `outputs/recovery_preflight_server/recovery_preflight_checks_final/{preflight.json,runtime.log,rejection.log,pytest.log,code_commit.txt}`. Local and server config/manifest fingerprints agree; the initial adapter remains the original real v2 adapter, not a new v3 training result.

Authorization remains the same genuine next gate across the original recovery-preparation turn and subsequent automatic continuations. The original requirement explicitly says to wait for confirmation before new long training. No actual user confirmation of the bounded v3 run has arrived; an automatic goal continuation is not that confirmation. No matching v3 training process or new training output was present in the authorized staged checkout, and its verification jobs are terminal. The original scope is unchanged and completion is unproven. Further learned-policy progress and the fresh trained-model comparison require that approval; do not launch, silently lower the objective or substitute more software checks for the experiment.
