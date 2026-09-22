# GenRecAgent

GenRecAgent is a modular generative recommendation agent scaffold for e-commerce and content recommendation. It combines explicit planning, dynamic tool routing, dense-retrieval and semantic-ID extension points, verification, and bounded re-planning.

The project is intentionally small in Phase 0: the runnable path is deterministic and local, so the architecture can be explained and tested before adding models, datasets, or online services.

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

Phase 0 is complete:

- independent Git project scaffold
- typed, serializable agent state
- deterministic planner, router, executor, and verifier loop
- mock filter, retrieval, and ranking tools
- LLM abstraction with a network-free OpenAI-compatible adapter stub
- recommendation and agent evaluation interfaces
- tests and lint configuration

Semantic ID, generative retrieval, SFT, RL, real datasets, and real experiment results are not implemented yet. They remain explicit roadmap items; this repository does not claim results that have not been measured.

## Project Structure

```text
configs/                 Runtime configuration placeholders
data/                    Ignored raw/processed data directories
docs/                    Architecture and roadmap notes
scripts/                 Safe local demo and future experiment entry points
src/genrecagent/         Package code
tests/                   Unit tests for the Phase 0 contracts
```

## Quick Start

```powershell
conda activate genrecagent
python -m pip install -r requirements.txt
python scripts/run_demo.py
python -m pytest
ruff check .
```

The demo uses no API key, model, dataset, or network request. To prepare a future provider configuration, copy `.env.example` to `.env`; never commit secrets.

## Evaluation Plan

The evaluation package reserves interfaces for Recall@K, NDCG@K, task success rate, constraint satisfaction, tool selection accuracy, invalid tool calls, average tool calls, average agent steps, and latency. Phase 0 only validates the software path. Real benchmark numbers should be added with reproducible datasets and experiment metadata in later phases.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the staged plan from a dense-retrieval baseline through semantic IDs, generative retrieval, dynamic routing, post-training, and ablations.

## References

The repository currently contains no copied model or dataset implementation. Relevant papers and system references should be added here as each research phase is implemented and verified.

