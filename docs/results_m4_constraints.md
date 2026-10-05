# M4 controlled intents and hard metadata price constraints

Run date: 2026-10-05 (Asia/Shanghai). The real-data router now accepts a bounded English/Chinese grammar for history recommendations, popularity recommendations and explicit code-transition recommendations. Price limits are applied before Dense/Popularity retrieval and retained through fallback; generated items are checked and filtered before acceptance. A final verifier checks the actual catalog metadata. Unknown, nonfinite, negative or nonnumeric prices cannot satisfy a budget. `under`/`below`/`低于` mean a strict upper bound; `at most`/`不超过` include the boundary.

This records the rule-based constraint increment. The subsequent actual local Qwen integration and curated paraphrase comparison are reported in [results_m4_llm_policy.md](results_m4_llm_policy.md).

The source catalog contains 112,590 items: **17,704 known prices and 94,886 missing prices**. Missing-price products remain eligible for requests without a budget. Price checks compare the numeric static source metadata; they do not establish current offers or perform currency conversion. Unsupported currencies, product-specific text searches, availability/vegan conditions and unrecognized compound requests receive an explicit rejection instead of silently losing conditions. The two broad category labels cannot support fine product-category constraints.

## Task protocol

`configs/m4_intent_tasks.json` defines 32 manually labeled templates. Each is combined with a real user's train-item history, a synthetic empty history, and a synthetic unknown-ID history, yielding **48 development and 48 held-out tasks**. Template IDs and user groups do not overlap across splits. User selection uses a deterministic SHA-256 ordering with seed 42; no validation/test next-item target is used to construct tasks. Each accepted task requests five items.

Per split there are 24 feasible requests (21 with a budget), three zero-budget requests with no eligible catalog item, nine malformed/conflicting-budget requests, and twelve unsupported requests. The task evaluator uses manual labels and raw item metadata, independently of the agent's intent parser and filtering helper. It checks price, unknown-price admission, seen/duplicate items, count, outcome status and bounded route trace. Returning an empty list for a feasible task fails the contract.

The held-out split is a **curated controlled-language software suite**, not a sample of real free-form user requests. Its perfect scores should be reported with that qualification. The four main task labels are `fulfilled`, `insufficient_candidates`, `invalid_request`, and `unsupported_request`; valid partial outputs do not claim that the requested count was fulfilled.

## Measured results

The comparison policy retrieves only the original top five and filters them afterward. It uses the same parser and final price checks, making it a diagnostic baseline for insufficient candidate retrieval, not a tuned recommendation-quality reference.

| Split | Policy | Parse correct | Output contracts passed | Feasible requests filled | Budget-feasible requests filled | Price violations / unknown-price admissions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Development | Filter before retrieval + verify | 48/48 | 48/48 | 24/24 | 21/21 | 0 / 0 |
| Development | Retrieve top five, then filter | 48/48 | 27/48 | 3/24 | 0/21 | 0 / 0 |
| Held out | Filter before retrieval + verify | 48/48 | 48/48 | 24/24 | 21/21 | 0 / 0 |
| Held out | Retrieve top five, then filter | 48/48 | 27/48 | 3/24 | 0/21 | 0 / 0 |

Both policies correctly rejected 21 invalid/unsupported requests per split and returned insufficient-candidate status on the three zero-budget requests. In the held-out split the new policy returned **105 budget-compliant items**, while the post-filter baseline returned 13 compliant items spread across incomplete requests. This demonstrates candidate completeness under the measured price constraints, not a Recall/NDCG or business uplift. The new policy's mean request latency on this task mix was 8.38 ms versus 1.35 ms for the post-filter policy; these include empty/rejected requests and cached eligibility sets, and are not deployment throughput measurements.

The original 8,151-user **unconstrained validation** protocol was rerun after adding eligibility support. All six Dense Recall/NDCG@5/10/20 values remained exactly equal to M1, including Recall@20 0.04686541528646792; all requests selected Dense with no failures. Mean route latency in this regression run was 5.58 ms. The recommendation test split was not reevaluated.

The full project regression suite passed **71 tests** and Ruff. Tests include strict price boundaries, missing/invalid metadata, retrieval beyond the original top-K, partial/empty safe outputs, price retention through Popularity fallback, and an adversarial grader check that rejects a fabricated displayed price when raw metadata is missing. The Phase 0 verifier was also corrected to reject unknown prices under a budget; its import path remains lightweight.

## Provenance and reproduction

The task specification, formal generation, development evaluation, held-out evaluation and unconstrained regression used clean code commit `8db202f5020f4850aa3c3c2d08d08b194c23ed01`. No policy change occurred between formal development and held-out evaluation. Task-file SHA-256: development `acfa8cfa1c5a88f052db14af06432ee639d13c654d7a18b54383975e67905fbb`; held out `d968f6d5d9ea959e99e283fed593435607456ea6b7b863ff12bc9a84cfb7f9f2`. The manifest records the template, prepared-history and item-file hashes; all evaluations verify those fingerprints.

Local ignored evidence:

- `data/processed/All_Beauty_m4_tasks/{development.jsonl,heldout.jsonl,manifest.json}`
- `outputs/m4_constraints_development/{metrics.json,run_config.json,traces.jsonl}`
- `outputs/m4_constraints_heldout/{metrics.json,run_config.json,traces.jsonl}`
- `outputs/m4_router_constraint_regression_validation/{metrics.json,run_config.json}`

Commands, with fresh output directories for reproductions:

```powershell
python scripts/prepare_m4_tasks.py --output-dir data/processed/All_Beauty_m4_tasks_reproduction
python scripts/evaluate_m4_constraints.py --tasks-dir data/processed/All_Beauty_m4_tasks_reproduction --split development --output-dir outputs/m4_constraints_dev_reproduction
python scripts/evaluate_m4_constraints.py --tasks-dir data/processed/All_Beauty_m4_tasks_reproduction --split heldout --output-dir outputs/m4_constraints_heldout_reproduction
python scripts/run_m4_constraints.py --text "根据我的历史推荐商品，价格不超过10美元。" --history-item B01DNS649G --top-k 3
```

The last command was exercised against the real catalog and returned three Dense items with metadata prices 8.09, 9.99 and 5.22. These commands use the existing local data and index. Broad free-form interpretation and provider-backed tool selection remain subsequent work; the current fixed grammar provides a typed, testable boundary for that policy.
