# GenRecAgent

GenRecAgent is a research project for personalized e-commerce recommendation with planned dense and generative retrieval plus a tool-using agent. The current repository contains a Phase 0 mock agent and a real-data M1 Popularity/Dense baseline on a documented All_Beauty development subset.

The mock agent is deterministic and local. Its catalog and trace are software checks, not recommendation benchmarks.

## Motivation

Modern recommendation systems increasingly need to understand natural-language intent, combine retrieval strategies, respect constraints, and recover from invalid intermediate results. GenRecAgent provides a clean research surface for those questions without coupling the core agent loop to a particular LLM, vector database, or training framework.

## Architecture

```text
UserRequest
    -> Planner
    -> Router
    -> Executor -> FilterTool / RetrievalTool / RankingTool
    -> Verifier
       | success -> final response
       | failure -> bounded re-plan
```

The tool contract is deliberately narrow (`name`, `description`, `run`). A future dense retriever, semantic-ID generator, reranker, or policy tool can be registered without rewriting the agent loop.

## Current Status

Implemented and locally verified:

- independent Git project scaffold
- typed, serializable agent state
- deterministic planner, router, executor, and verifier loop
- mock filter, retrieval, and ranking tools
- LLM abstraction with a network-free OpenAI-compatible adapter stub
- recommendation and agent evaluation interfaces
- tests and lint configuration
- M1 typed review/item data handling, chronological leave-last-out, and deterministic user sampling on both local fixtures and official All_Beauty data
- BGE-encoded full item catalog, NumPy exact cosine/IP retrieval, saved item mapping, train-only popularity, and validation/test evaluation on a 5,000-user real-data subset

Implemented but not yet verified: optional FAISS IndexFlatIP backend. The exact NumPy backend has been used for the reported M1 result.

Planned: trained RQ-VAE Semantic IDs, trained Transformer generative retrieval, real Qwen tool policy, SFT, and unified ablations. M1's **development-subset real-data baseline is complete**; a full-eligible-user run remains open.

## Project Structure

```text
configs/                 Runtime configuration placeholders
data/                    Ignored raw/processed data directories
docs/                    Architecture and roadmap notes
scripts/                 Safe local demo and future experiment entry points
src/genrecagent/         Package code
tests/                   Phase 0 and M1 unit/integration tests
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

The real 5,000-user development-subset test run (21,465 interactions, seed 42, full 112,590-item catalog, BGE small English v1.5) produced Recall@20 of **0.0086 Popularity** and **0.0368 Dense**; NDCG@20 was **0.00370** and **0.02099** respectively. These are offline next-item results. See [docs/results_m1.md](docs/results_m1.md) for both splits, all K values, hashes, exact model revision, latency scope and limitations.

Agent metrics remain software-level placeholders until a separate reproducible natural-language task set exists. See [docs/implementation_plan.md](docs/implementation_plan.md) for milestones and evidence status.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the staged plan from a dense-retrieval baseline through semantic IDs, generative retrieval, dynamic routing, post-training, and ablations.

## References

The repository currently contains no copied model or dataset implementation. Relevant papers and system references should be added here as each research phase is implemented and verified.
