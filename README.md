# GenRecAgent

GenRecAgent is a research project for personalized e-commerce recommendation with dense and generative retrieval and bounded real-tool execution. It includes measured Popularity/Dense baselines, a separate residual K-Means Semantic ID baseline after an RQ-VAE collapse, a fully evaluated small Transformer, actual item-level dual retrieval, and fixed/Prompt/SFT multi-step policies with verified termination. **Real two-epoch recovery Qwen LoRA training and a fresh 72-task paired test are complete**, with weights and logs returned from lab95. Raw recovery SFT passed 72/72 versus same-input Prompt 0/72 and Fixed 72/72, without a runtime controller. This is a synthetic normalized-state tool-policy result; learned next-item retrieval remains weak and no generative recommendation gain is claimed.

The mock agent is deterministic and local. Its catalog and trace are software checks, not recommendation benchmarks.

M2 repair update (2026-10-09): the approved **learned PCA-warmstarted EMA RQ-VAE** completed full-catalog training on lab95, using all 112,590 items and the original item split. Selected-validation and full-catalog frozen gates both pass: catalog usage 256/256/256, 93,350 unique codes and validation MSE 0.00041323. Independent CUDA re-encoding finds zero mismatches over the entire catalog; all checksummed artifacts are local. The full software suite passes 181 tests. The original collapsed model and separate RQ-KMeans baseline are preserved. **Current M3 still uses its original K-Means map/checkpoint**; migration requires fresh prefixes and separately approved training. See [actual full results and limitations](docs/results_m2.md) and [the frozen plan](docs/m2_training_plan.md).

M3 migration preparation (2026-10-09): new RQ-VAE-coded samples are independently materialized under `data/processed/All_Beauty_m3_rqvae_stable/` (10,969 train / 8,151 validation / 8,151 test). All 27,271 rows pass independent chronological/code checks and match the original item/user protocol. **190 software tests pass**. Separate config `configs/generative_rqvae_stable.yaml` pins inputs and requires explicit training approval; no new full training/evaluation has started and old delivery defaults stay unchanged. See [the preparation evidence and bounded approval proposal](docs/m3_rqvae_stable_plan.md).

Latest development: full fixed-weight retrieval test is complete (8,151 users, zero source failures). The user approved recovery v3 training on 2026-10-08; the job completed with exit 0 and its checksummed artifacts are local in `outputs/tool_sft_recovery_v3_server`. All 112 LoRA tensors changed and remained finite. Full raw validation passed 70/72, with two step-budget failures retained; fresh raw test passed 72/72 with one rejected model call recovered. Independent grading checked all 288 validation/test executions. Read the [actual recovery run and limits](docs/recovery_v3_plan.md) and [original-scope audit](docs/acceptance_audit_2026_10_07.md). No controller success is counted as raw model capability.

## Start here — real-data delivery entry

On this machine, the prepared data, retrieval checkpoint and returned SFT adapter are already present:

```powershell
conda activate genrecagent
$env:PYTHONNOUSERSITE = "1"
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
python scripts/run_project.py check
python scripts/run_project.py demo
python scripts/run_project.py report
```

`check` reports missing files/dependencies/CUDA without downloading anything. `demo` runs real-catalog hybrid, cold-start and infeasible-budget scenarios, defaulting to Fixed. `report` aggregates existing frozen experiments into Markdown/JSON, keeping retrieval test results, validation-only fusion and synthetic policy task metrics separate. Fresh timestamped output directories are selected automatically; previous runs are never overwritten. The report command does not require CUDA, but needs the existing evidence and trained adapter for checksum verification.

For the opt-in trained policy: `python scripts/run_project.py demo --policy sft --scenario hybrid`. To verify all cached Qwen files: `python scripts/run_project.py check --policy sft --verify-hashes`. These commands perform checks/inference only, not training. Heavy artifacts are ignored by Git, so a fresh public clone is not automatically demo-ready.

For the **new raw recovery model**, use `python scripts/run_project.py demo --policy compact_sft`. The default adapter is the returned v3 checkpoint; legacy `sft` still uses the old v2 adapter. `python scripts/run_project.py report` independently verifies the v3 training, frozen selection, finite actual tensor updates and complete paired traces before reporting its results. Fixed remains the fastest default; neither report nor demo retrains or downloads models. Windows native Torch instability is a separate known environment issue; the same CLI is exercised on stable lab95.

The current local delivery outputs are `outputs/project_delivery/demo/` and `outputs/project_delivery/report/`. Read the [Chinese usage and project explanation](docs/project_delivery_zh.md) or the [actual SFT evidence](docs/results_m5_sft_lab95.md). The old `run_demo.py` entry below is explicitly the Phase 0 mock, not this real-data demo.

For a lightweight local reviewer snapshot, run `python scripts/package_project.py` from a clean committed working tree. It creates a ZIP and self-contained HTML replay of the saved real-tool demos, with no weights, raw data, logs or Git history. It does not upload or publish anything. See [the sharing scope](docs/share_package_zh.md) and [resume/presentation draft](docs/resume_project_zh.md). The replay is explicitly archived execution, not live inference or a new benchmark.

## Motivation

Modern recommendation systems increasingly need to understand natural-language intent, combine retrieval strategies, respect constraints, and recover from invalid intermediate results. GenRecAgent provides a clean research surface for those questions without coupling the core agent loop to a particular LLM, vector database, or training framework.

## Architecture

```text
Text + observed history + explicit constraints
    -> intent/constraint normalization
    -> next-action policy: Fixed / raw Qwen Prompt / Qwen + LoRA
    -> ToolEnvironment -> real Dense / Transformer / Popularity retrieval
    -> filter -> rank -> verify
       | verified and sufficient -> finish
       | failure/shortfall -> bounded recovery or clarify
```

The diagram describes the real-data path, not the Phase 0 mock. `scripts/run_tool_agent.py` implements one request; `scripts/run_project.py` wraps readiness, preset demonstrations and evidence aggregation. Fixed orchestration and raw Qwen policies are evaluated separately; no hidden Fixed fallback is counted as raw SFT success. State constraints remain immutable throughout the ten-action budget. See [the current SFT report](docs/results_m5_sft_lab95.md).

## Current Status

Implemented and exercised locally or on lab95 (model quality caveats remain):

- independent Git project scaffold
- typed, serializable agent state
- deterministic planner, router, executor, and verifier loop
- mock filter, retrieval, and ranking tools
- LLM abstraction with a network-free OpenAI-compatible adapter stub
- recommendation and agent evaluation interfaces
- tests and lint configuration
- M1 typed review/item data handling, chronological leave-last-out, and deterministic user sampling on both local fixtures and official All_Beauty data
- BGE-encoded full item catalog, NumPy exact cosine/IP retrieval, saved item mapping, train-only popularity, and validation/test evaluation on a 5,000-user real-data subset
- M2 RQ-VAE training, checkpointing and collision-aware mapping on all 112,590 items; original collapsed run retained (1,718 codes), stabilized learned full run now passes frozen tokenizer gates (93,350 codes)
- Separate PCA-whitened residual MiniBatch K-Means baseline on the same item vectors and item split; 97,231 unique three-token codes, with all remaining collisions recorded
- M3 leakage-aware prefix examples, catalog-valid Semantic ID trie decoding, and collision-aware item candidate expansion tested on the full catalog with deterministic mock scores
- M3 train-only code-transition baseline and a full-trained causal Transformer, both evaluated on all 8,151 validation and test users
- M4 dense-first offline router over the real catalog, with train-only Popularity cold-start/fallback and an opt-in code-transition route; full-validation metrics exactly reproduce M1 Dense
- M4 controlled intents and hard price constraints, with 48 development/48 held-out synthetic tasks, independent metadata grading and zero measured price violations
- Local Qwen2.5-1.5B JSON tool decisions with strict schema checks, grounded budgets and one repair; curated paraphrase held-out contracts improved from rules 9/30 to guarded pipeline 27/30, while raw model labels were only 3/10 unique expressions correct
- Actual Dense + trained M3 Transformer item-level rank fusion, evaluated on all 8,151 validation users; equal weighting hurts recall, while 75% Dense weighting does not improve Recall@20
- The frozen full test reproduces original Dense/Transformer endpoints; 75% Dense fusion Recall@20 is 0.036928 versus Dense 0.036560 (only three additional net hits, no robust-gain claim)
- Complete artifact-only candidate pools, explicit end reasons and non-vacuous metric denominators; an offline evidence checker verifies frozen inputs, candidate filtering/provenance and summary grades without inference
- A nine-tool stateful environment with real catalog execution, immutable observations, bounded recovery and verified termination; fixed-flow and raw multi-step Qwen comparisons plus recovery/finish-gate ablations
- Executed v2 trajectories (608 train / 492 validation / 490 test assistant actions), user-disjoint splits, assistant-only labels and a local LoRA entry verified on a tiny random CPU software test
- Actual Qwen2.5-1.5B LoRA training on lab95 RTX 3090: validation action NLL 0.6410 to 0.0382; same 12 held-out task contracts Fixed 12/12, Prompt 0/12, raw SFT 4/12 (only one of nine feasible tasks filled). Adapter hashes, returned artifacts and limitations are in [the SFT report](docs/results_m5_sft_lab95.md)

Implemented but not yet verified: optional FAISS IndexFlatIP backend. The M1 result used exact NumPy search.

The minimum experimental chain now includes actual SFT, paired policy comparison, retrieval comparison, verification/recovery ablations and CLI traces. Keep Fixed as the reliable demonstration policy; raw SFT remains an opt-in experiment. Larger independent language sets, better learned policies and public-release packaging can follow. M2's stabilized learned tokenizer now meets its minimum noncollapse gates, but the existing M3/delivery runtime still uses the separately named **RQ-KMeans baseline**. New RQ-VAE codes have not been evaluated for next-item retrieval. This checkpoint is not a production-ready learned recommender.

## Current development — guarded model runtime

`guarded_sft` and `guarded_prompt` gate model proposals by observable tool prerequisites and immutable request constraints. Rejected proposals are replaced by explicitly logged controller actions; after two rejections the controller completes the request. Raw `sft`/`prompt` modes remain available and their historical results are unchanged. In a fresh **72-task validation-only** comparison, raw SFT passed 26/72 and guarded SFT passed 72/72, but every guarded task used the controller and zero were unassisted. This improves system execution, not the learned model's independent ability. Fixed remains the lower-cost default.

```powershell
python scripts/run_project.py demo --policy guarded_sft
python scripts/evaluate_tool_agent.py --policy guarded_sft --split validation --max-tasks 72 --output-dir outputs/new_guarded_validation
```

These commands use the existing lab95-trained adapter locally for inference only. New training, if authorized, still runs on lab95. See [runtime validation evidence](docs/results_guarded_policy_validation.md).

## Project Structure

```text
configs/                 Retrieval, model and SFT configurations
data/                    Ignored raw/processed data directories
docs/                    Architecture and roadmap notes
scripts/                 Delivery, data preparation, training and evaluation entries
src/genrecagent/         Package code
tests/                   Phase 0 through SFT/delivery unit/integration tests
outputs/                 Ignored experiment records
artifacts/               Ignored local model cache
```

## Quick Start

```powershell
conda activate genrecagent
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python scripts/run_demo.py
python -m pytest -p no:anyio --basetemp outputs/pytest_tmp
ruff check .
```

The demo uses no API key, model, dataset, or network request. On this Windows host, `-p no:anyio` avoids an unrelated user-level pytest plugin. To prepare a future provider configuration, copy `.env.example` to `.env`; never commit secrets.

## Dataset and M1 baseline

The data source is [Amazon Reviews 2023 / All_Beauty](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). Reviews are observed interactions; they are not necessarily purchases. Both review and metadata records use `parent_asin` as the item ID. At revision `2b6d039`, the official repository has a raw review JSONL file and a metadata Parquet file for All_Beauty. The loader streams these directly without executing the old remote loading script. `scripts/prepare_amazon.py --download` is an explicit action and stores the files under `data/raw/hf_cache`.

Invalid review IDs/timestamps and reviews without metadata are discarded. Multiple reviews by one user of the same parent item are reduced to the earliest valid event. Equal timestamps use source row order, which is deterministic but does not reveal the true within-time order. Missing prices remain missing; static metadata price is not a live offer. Users need at least three valid distinct items. The final item is test, the previous item validation, and earlier items train. Validation history is train; test history is train plus the now-observed validation item. Popularity uses train interactions only.

The default candidate universe is the full retained metadata catalog, including items absent from train interactions. This assumes static catalog visibility in an offline experiment, not historical product availability. User sampling uses SHA256 of seed and user ID, independent of hit outcomes. The optional `--max-items` mode samples catalog IDs independently of targets and reports unreachable targets. Per-user ordering alone does not guarantee cross-user global temporal separation.

```powershell
# Requires the official data and model/runtime footprint:
python -m pip install -r requirements-encoder.txt
python scripts/prepare_amazon.py --category All_Beauty --download --revision 2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e --max-users 5000
python scripts/build_index.py --device cuda --allow-model-download
python scripts/evaluate_dense.py --split both --output-dir outputs/m1_reproduction
python scripts/run_dense_demo.py --query "moisturizer for dry skin" --top-k 5
```

The tested GPU runtime on this machine was `torch==2.6.0+cu124` from the [official PyTorch CUDA 12.4 index](https://download.pytorch.org/whl/cu124), selected to match the installed driver. `requirements-encoder.txt` alone may install a CPU wheel; check `torch.cuda.is_available()` and select an appropriate official CUDA build for your machine. For files already obtained from the official source, pass `--review-file` and `--metadata-file`, plus `--source-kind official_local` and a known `--revision` if available. Outputs are refused when the target directory already contains files. Use fresh preparation, index and evaluation output directories for another run. The text-query demo embeds a query string and retrieves products; `evaluate_dense.py` pools observed user-item vectors to predict the next item. They are different tasks.

## Evaluation Plan

For M1, `evaluate_dense.py` reports popularity and dense Recall@5/10/20 and NDCG@5/10/20 on validation and test, with one target per user and ranks starting at 1. The same catalog and seen-item policy are used for both methods. Missing targets remain in the denominator and are counted. Retrieval latency includes user pooling and search, excludes model loading, and is machine-specific. The run saves `metrics.json` and `run_config.json` under ignored `outputs/`. Fixture numbers are never reported as real results.

The primary full-eligible-user test run (8,151 users, 35,422 interactions, full 112,590-item catalog, BGE small English v1.5) produced Recall@20 of **0.006625 Popularity** and **0.036560 Dense**; NDCG@20 was **0.002784** and **0.021247** respectively. These are offline next-review-item results. See [docs/results_m1_full.md](docs/results_m1_full.md) for both splits, all K values, hashes, exact model revision, latency scope and limitations. The separate 5,000-user development result is in [docs/results_m1.md](docs/results_m1.md).

## M2 Semantic ID training (full tokenizer gates passed)

`scripts/train_rqvae.py` reads the saved M1 item embeddings, splits items by a fixed seed for reconstruction training/validation, trains a residual-quantized autoencoder, selects the checkpoint by validation reconstruction MSE, and saves learned codebooks, an item-to-token mapping, usage and collision statistics. A code maps to **all** colliding catalog items; it is not silently resolved using held-out targets.

The CPU smoke command below trains only 128 items for two epochs and is a software check:

```powershell
python scripts/train_rqvae.py --smoke-test --max-items 128 --epochs 2 --batch-size 32 --hidden-dim 64 --latent-dim 16 --num-levels 2 --codebook-size 16 --device cpu --output-dir outputs/m2_cpu_smoke_new
```

The original full-catalog RQ-VAE confirmed severe collapse despite a valid checkpoint. A separate PCA-whitened residual MiniBatch K-Means baseline yielded 97,231 unique codes and 24,794 colliding products. The 2026-10-09 approved stabilized neural RQ-VAE now yields 93,350 unique codes and 30,055 colliding products, with both predeclared quality gates passed and full-catalog reload/encoding verified. Its reconstruction improves, but its collision count is worse than K-Means; no overall superiority or next-item gain is claimed. All three runs share the exact input and item split and remain separately preserved in [docs/results_m2.md](docs/results_m2.md). Returned new artifacts: `outputs/m2_rqvae_stable_full_server/m2_rqvae_stable_full/`; do not substitute this map into the old M3 checkpoint.

## M3 Generative Retrieval

`python scripts/prepare_m3_sequences.py` prepares user-history-to-next-code examples using only train-period prefixes for supervised training. On the full eligible-user data it produces **10,969 train**, **8,151 validation** and **8,151 test** examples. The valid-code trie masks impossible token paths; the candidate adapter expands colliding codes, filters seen items and uses train-only popularity for deterministic within-code ordering. See [docs/m3_protocol.md](docs/m3_protocol.md) for exact split, scoring and collision rules.

The train-only code-transition baseline ran on all 8,151 validation and test users. Test Recall@20 was **0.007852**, below Dense **0.036560** under the same item universe; see [docs/results_m3_transition.md](docs/results_m3_transition.md). A 473,731-parameter causal Transformer was subsequently trained on all 10,969 train prefixes, selected by validation target-token loss, and evaluated with the same full-catalog constrained decoder. Its test Recall@20 was only **0.002454** despite zero invalid generated codes or candidate shortfalls. This is a negative model result, not a generative-retrieval win; see [docs/results_m3_transformer.md](docs/results_m3_transformer.md) for provenance, all metrics and limitations. The earlier resource plan remains in [docs/m3_transformer_plan.md](docs/m3_transformer_plan.md).

A validation-only follow-up found that missing the correct Semantic ID code, rather than invalid decoding or collision expansion, dominates the Transformer failure. A fixed code-rank fusion grid added only one validation hit over the count baseline at its best weight, so it was **not** promoted to a test run. See [docs/results_m3_validation_diagnostics.md](docs/results_m3_validation_diagnostics.md) for cohorts, failure counts, fusion ablation and the next modeling decision.

Recommendation Recall/NDCG and synthetic language/tool task contracts are different evaluations. Do not treat task-contract success as recommendation relevance. See [docs/implementation_plan.md](docs/implementation_plan.md) for milestones and evidence status.

## M4 Dense-first offline routing

The real-data history router selects Dense by default, uses train-only Popularity for cold starts or invalid Dense results, and permits code-transition generation only through an explicit experimental mode. It verifies catalog membership, seen/duplicate items, ranks, scores and candidate count, with a bounded attempt trace. All 8,151 default-mode validation users selected Dense with zero failures, and every Recall/NDCG@5/10/20 value exactly matched the original M1 Dense result. In a separate explicit-generative validation run, three of 8,151 requests fell back to Dense because generated candidates were insufficient. See [docs/results_m4_router.md](docs/results_m4_router.md) for the first increment.

```powershell
python scripts/run_m4_router.py --history-item B01DNS649G --top-k 3
python scripts/run_m4_router.py --top-k 3
python scripts/evaluate_m4_router.py --mode auto --output-dir outputs/m4_router_validation_reproduction
```

The first command traces Dense routing, the second traces Popularity cold start, and the third evaluates the validation split only. Use `--mode generative` with the trace command to opt into the weaker code-transition route and inspect its fallback behavior. These commands require the local prepared data/index and do not download a model.

The subsequent constrained agent accepts fixed English/Chinese phrases for history, popular and generative recommendations with optional maximum price. It excludes unknown prices when a budget is present, preserves eligibility through fallback, and reports insufficient candidates or unsupported conditions explicitly. A 48-task held-out synthetic suite passed all output contracts; 21/21 budget-feasible requests were filled versus 0/21 for a top-five-then-filter diagnostic baseline. These are curated software task results, not broad natural-language or recommendation-relevance scores. See [docs/results_m4_constraints.md](docs/results_m4_constraints.md).

```powershell
python scripts/run_m4_constraints.py --text "根据我的历史推荐商品，价格不超过10美元。" --history-item B01DNS649G --top-k 3
```

## Local Qwen tool-policy extension

The pinned Qwen2.5-1.5B-Instruct snapshot is cached locally. `run_llm_policy.py` compiles a request into checked JSON tool arguments and uses the same catalog verifier. Unknown tools, malformed/null fields and ungrounded budgets receive one repair attempt, then conservative rule fallback. On held-out curated paraphrases, the full pipeline passed 27/30 contracts versus rules 9/30, with zero observed budget violations; the raw model first-call labels were correct on only 3/10 independent expressions. The original controlled suite retained 48/48 pipeline contracts. See [docs/results_m4_llm_policy.md](docs/results_m4_llm_policy.md) for both splits, raw failures, timing and limits.

```powershell
$env:PYTHONNOUSERSITE = "1"
python scripts/run_llm_policy.py --text 'Please base your picks on my past interactions; each item should be no more than $20.' --history-item B01DNS649G --top-k 3
```

This explicit model entry adds seconds of inference cost; the existing rule entry remains available. Runtime requirements are in `requirements-policy.txt`; the separate `download_qwen_policy.py` pins revision/file hashes when the snapshot is absent. Model inference is local and does not require a provider API key.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the staged plan from a dense-retrieval baseline through semantic IDs, generative retrieval, dynamic routing, post-training, and ablations.

## Minimal completion path

Use the existing data, index and locally cached models; no new model download is needed on the development machine.

```powershell
conda activate genrecagent
$env:PYTHONNOUSERSITE = "1"
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:HF_HUB_OFFLINE = "1"
python scripts/run_tool_agent.py --policy fixed --mode hybrid --history-item B01DNS649G --max-price 20 --output outputs/my_tool_demo.json
python scripts/evaluate_tool_agent.py --policy fixed --output-dir outputs/my_fixed_comparison
```

Load the adapter already trained on lab95 (local inference only):

```powershell
python scripts/run_tool_agent.py --policy sft --adapter outputs/tool_sft_lab95_server/tool_sft_lora/adapter --mode hybrid --history-item B01DNS649G --max-price 20 --output outputs/my_sft_demo.json
```

For a separately approved reproduction on lab95:

```powershell
python -m pip install -r requirements-sft.txt
python scripts/train_tool_sft.py --output-dir outputs/my_tool_sft
python scripts/evaluate_tool_agent.py --policy sft --adapter outputs/my_tool_sft/adapter --output-dir outputs/my_sft_comparison
```

The configured full run uses Qwen2.5-1.5B, q/v LoRA rank 8, one epoch and 1,024-token samples. A `--smoke-test` run uses a tiny random CPU model and is **not** policy SFT; evaluation refuses that adapter. All output paths must be fresh. The pre-SFT checkpoint is in [docs/results_fast_completion.md](docs/results_fast_completion.md), and the completed training, final same-machine table and returned artifact locations are in [docs/results_m5_sft_lab95.md](docs/results_m5_sft_lab95.md).

## References

The repository currently contains no copied model or dataset implementation. Relevant papers and system references should be added here as each research phase is implemented and verified.
