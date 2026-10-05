# M4 local Qwen tool policy: raw decisions, guards and verified execution

Run date: 2026-10-05 (Asia/Shanghai). A real local **Qwen2.5-1.5B-Instruct** model now produces a JSON function call selecting personalized, popular or explicitly requested generative recommendations and an optional price limit. The pretrained model was used for inference without weight training. It is connected to the existing real catalog, Dense/code-transition/Popularity routes and metadata price verifier. The original rule-based entry point remains available and is not replaced by this weaker raw model policy.

## Model and execution boundary

Model repository `Qwen/Qwen2.5-1.5B-Instruct`, fixed revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; weights are 3,087,467,144 bytes with SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`. The tokenizer/config/license files and their hashes are retained in `outputs/m4_qwen_download/manifest.json`. The model's official card identifies the Apache-2.0 license and provides its chat-template interface. [Official Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

Inference ran locally on RTX 4060 Laptop GPU using float16, PyTorch `2.6.0+cu124` and Transformers `5.17.0`, with greedy decoding (`do_sample=false`), at most 2,048 input tokens and 192 output tokens per attempt. Model loading is local-only after the separate pinned download. This is prompted JSON function calling through the provider-neutral `BaseLLM` seam, not an external API request.

`recommendation_policy` accepts only the declared status, intent, price limit and reason fields. Unknown function names, extra arguments, duplicate JSON keys, nonfinite numbers, bad types and incomplete budget evidence are rejected. Before execution, deterministic guards compare budget amount/boundary and a verbatim evidence span with the request, reject recognized unsupported currencies/conditions, reject combined popular+generative routes, require an explicit generative request and enforce agreement with already-recognized rule grammar. These guards cover the documented patterns; they do not prove correct understanding of every arbitrary free-form condition.

An invalid call receives one repair attempt with schema errors and budget observations derived from the **request text**, not task labels. A second failure uses the conservative rule parser. Raw responses, decoded calls, schema/guard errors, repair attempts, token counts, cache flags and both policy/retrieval fallback traces are saved. Candidate retrieval and final price verification remain deterministic. The model never receives expected labels, user IDs, held-out next-item targets or catalog candidate IDs, and cannot directly fabricate final items.

## Task sets and development boundary

- Original controlled suite: 48 development / 48 held-out tasks, with 16 distinct expressions per split.
- New manually curated paraphrases: 30 development / 30 held-out tasks, with 10 distinct English/Chinese expressions per split. Each expression is combined with known train-item history, empty history and unknown-ID history. Template/user groups are disjoint within each split pair; seed 73 selects train-only contexts for the new suite.

These are synthetic software task suites on the real 112,590-item catalog, not a broad natural-language benchmark. Three contexts sharing one expression are **not three independent language examples**. Intent decisions are cached by text across contexts, with that reuse explicitly counted. The independent grader checks manual labels, raw metadata, output count, price/seen-item rules and rejection status.

Development exposed fabricated evidence spans, missing/null fields and an incorrectly accepted popular+generative compound request. Earlier development evidence is preserved in `outputs/m4_llm_control_development/` (45/48 contracts) and `outputs/m4_llm_freeform_development/` (18/30). Examples/schema descriptions, grounded repair feedback and a deterministic route-conflict check were refined using development/smoke results only. The final policy was frozen at commit `c3a0878747c6b775cfca234338f18862d779c4ab` before either held-out suite ran. No policy or prompt was tuned from held-out results.

## Final task results

| Suite / split | Rules contracts | Qwen + guard/repair/fallback contracts | Feasible requests filled: rules → guarded | First raw model labels correct, unique expressions |
| --- | ---: | ---: | ---: | ---: |
| Controlled development | 48/48 | 48/48 | 24/24 → 24/24 | 8/16 |
| Controlled held out | 48/48 | 48/48 | 24/24 → 24/24 | 8/16 |
| Paraphrase development | 9/30 | 27/30 | 0/21 → 18/21 | 4/10 |
| Paraphrase held out | 9/30 | 27/30 | 0/21 → 18/21 | 3/10 |

On controlled held-out tasks, the pipeline returned 105 budget-compliant items and filled 21/21 budget-feasible requests. On paraphrase held-out tasks, it returned 75 budget-compliant items and filled 15/18 budget-feasible requests; rules filled none. Both held-out runs had **zero observed price violations and zero unknown-price admissions under a budget**. These are task-coverage/contract results, not next-item Recall/NDCG or business gains.

First-call schema validity was 13/16 unique controlled expressions and 8/10 paraphrases. First raw label correctness is much lower than final pipeline success, and even that label metric does not certify the evidence span. Controlled held-out inference used 24 model calls for 16 expressions, eight repaired expressions and three rule-fallback expressions (nine tasks). Paraphrase inference used 17 calls for 10 expressions, seven repaired expressions and one rule-fallback expression (three tasks). Schema/provider errors totaled three and two; schema-valid calls rejected by guards totaled eight and six. Retrieval fallback counts are separate from these policy fallback counts.

The three paraphrase failures are all `fh10` across history contexts: a request includes an instruction to emit `price_limit=null` while actually asking for products below $10. Both model attempts omitted/changed the budget, the guard rejected them, and the conservative fallback returned unsupported with no items. Budget safety was preserved, but the expected request was not fulfilled. This single curated adversarial example does not establish general prompt-injection resistance.

## Cost, reproduction and interpretation

After separating independent grading from request timing, verified held-out generation totaled 49.50 seconds for 16 controlled expressions and 33.39 seconds for 10 paraphrases: approximately **3.09 / 3.34 seconds per uncached expression**, including repair generation. Full task mean request latencies were 1,034 / 1,124 ms because two-thirds of contexts reused a cached text decision. They must not be presented as uncached serving latency. Independent grading averaged about 26 / 34 ms and is excluded from the corrected request measurement. Greedy decoding and low raw reliability are recorded explicitly.

The initial evaluations accidentally included independent grading in the request stopwatch. Instrumentation-only commit `fe8f42e49aea810d942b4fe5262fd9e7a6742ede` corrected that scope. Both held-out suites were replayed with **unchanged model/prompt/schema/guard parameters**; contract counts, model-call counts and input/output token counts reproduced exactly. The verified replay outputs are the reported timing evidence. All runs had clean source revisions. Final prompt SHA-256 is `5bf68cf335db1ae2cd2d74510c2e11cc896eedb599cfb5ad9797a305c22ad7fe`; schema SHA-256 is `8c4064a17c3b48adda8b97be6d72994e34ca2957e03975295279c7d5515b6472`. Source/task/model hashes are included in each `run_config.json`.

The full project suite passed **83 tests** and Ruff. The complete demo was exercised with a development paraphrase and observed item `B01DNS649G`; Qwen emitted a grounded $20 inclusive limit in one call, and verified Dense retrieval returned three items priced 8.09, 10.95 and 12.60 in static metadata.

Ignored local evidence:

- `outputs/m4_llm_control_development_final/` and `outputs/m4_llm_freeform_development_final/`
- `outputs/m4_llm_control_heldout_verified/` and `outputs/m4_llm_freeform_heldout_verified/`, each with metrics, provenance and raw traces
- `data/processed/All_Beauty_m4_freeform/{development.jsonl,heldout.jsonl,manifest.json}`
- `outputs/m4_qwen_download/manifest.json` and the pinned snapshot under `artifacts/model_cache/`

```powershell
conda activate genrecagent
$env:PYTHONNOUSERSITE = "1"
python scripts/run_llm_policy.py --text 'Please base your picks on my past interactions; each item should be no more than $20.' --history-item B01DNS649G --top-k 3
python scripts/evaluate_llm_policy.py --tasks-dir data/processed/All_Beauty_m4_freeform --split development --output-dir outputs/m4_llm_freeform_dev_reproduction
```

Use fresh output directories. Existing `run_m4_constraints.py` remains the fast fixed-grammar route; `run_llm_policy.py` explicitly selects the local model path. The measured policy expands curated paraphrase coverage while adding seconds of cost and retaining substantial raw errors. More independent labeled expressions and verified trajectories are needed before fine-tuning or claiming general deployment reliability.
