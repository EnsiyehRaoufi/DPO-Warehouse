# warehouse short-order assistant - DPO pipeline

Loads a warehouse dataset, builds some tool functions
over it, has an assistant answer questions using those tools, makes
grounded vs broken versions of the answers, scores them with a rubric to
get DPO pairs, trains a small model with DPO (LoRA), and evaluates
before/after.

## reports and docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) - how the pipeline fits together,
  from raw data through to a trained model and a live answer to a question.
- [`RUBRIC.md`](RUBRIC.md) - the full 7-dimension scoring rubric used to
  build DPO preference pairs and to evaluate the trained model.
- [`assets/PROJECT_EVALUATION_REPORT.md`](assets/PROJECT_EVALUATION_REPORT.md) -
  project history, base model selection, environment setup, and the final
  evaluation results.

## about the dataset

source dataset: [`electricsheepafrica/warehouse-inventory-management`](https://huggingface.co/datasets/electricsheepafrica/warehouse-inventory-management/tree/main/data)


two datasets derived from it are published on Hugging Face:
[`EnRaoufi/warehouse-inventory-stratified-sample`](https://huggingface.co/datasets/EnRaoufi/warehouse-inventory-stratified-sample)
(the curated sample `build_sample.py` produces) and
[`EnRaoufi/warehouse-dpo-preference-pairs`](https://huggingface.co/datasets/EnRaoufi/warehouse-dpo-preference-pairs)
(the DPO `{prompt, chosen, rejected}` pairs `gen_pairs.py` produces from it).

## files

- `data_utils.py` - loads the data, has the tool functions (get_stockouts,
  get_backorders, rank_shortage_risk, get_warehouse_kpis, explain_inventory_issue)
- `assistant.py` - calls the tools, formats FACTS/RECOMMENDATION answers,
  makes broken versions for the rejected side of DPO pairs, AND has the
  `route()` function that maps a raw question to a tool + scope (see
  "routing" section below)
- `gen_pairs.py` - builds the prompt list (100+), generates pairs, has the
  scoring rubric, picks chosen/rejected
- `train.py` - DPO training w/ TRL + LoRA
- `eval_model.py` - runs base model and dpo model on the same held out
  prompts, dumps a report (also runs tool-selection accuracy separately)
- `run.py` - runs the whole thing
- `ask.py` - live demo: `python ask.py "any active stockouts?"` - actually
  routes a raw question and answers it, no pre-known intent
- `compare_models.py` - zero-shot comparison of candidate base models
  BEFORE committing to one for DPO training (see "picking a base model"
  below)
- `data/warehouse_sample.jsonl` - a 210-row real subset
  (pulled straight from the actual CSVs, with coverage of every issue type
  and all 3 warehouse levels). 
- `build_sample.py` - regenerates warehouse_sample.jsonl from the real 3
  CSVs (`python build_sample.py --data-dir data`). 

## routing (how the "LLM" picks the right tool)

`assistant.route()` is a **plain keyword classifier**, not the model doing real function-calling.
It looks at the raw question text for words like "backorder", "stockout",
"shortage"/"risk", pulls out a warehouse id via regex (`WH_\d+`) and a
commodity category by checking which real category names from the data
appear in the text, and picks one of the 5 required tools based on that.

## picking a base model

`base_model` (default `Qwen/Qwen2.5-0.5B-Instruct`) was picked as a
reasonable-sounding default - small, open, instruction-tuned, works
cleanly with TRL/PEFT. To verify it:

```
!python compare_models.py
```

full run (needs the 3 real CSVs dropped into `data/` first - see "about the
dataset" above for exact filenames, or point `--data-dir` at wherever you
put them):

```
python run.py --base-model Qwen/Qwen2.5-0.5B-Instruct --min-prompts 150 --data-dir data
```

or run against the bundled 210-row sample instead - no CSV download
needed, useful for a end-to-end check of the whole pipeline (this is done in this work):

```
!python run.py --sample --data-dir data --base-model Qwen/Qwen2.5-0.5B-Instruct --min-prompts 150 --out out --max-new-tokens 100
```

this writes to `out/`:
- `records.jsonl` - the normalized data
- `dpo_pairs.jsonl` - the DPO dataset (prompt/chosen/rejected)
- `eval_pairs.jsonl` - held out prompts, never used for training. has real
  `chosen`/`rejected` fields for the eval set as well as the training set. 
  also carries `tool_out`/`scope`/`uncertain` alongside
  those, since `eval_model.py` needs that ground truth to score a model's
  own freshly-generated text, not the pre-built candidates.
- `model/` - the trained adapter + `run_info.json` (config + dataset
  fingerprint) + `metrics.json` (training curve)
- `eval_report.json` - before/after comparison: baseline_score,
  post_dpo_score, delta, win_rate_vs_baseline, hallucination rate
  before/after, factual correctness, tool use correctness, uncertainty
  handling, top improvements/regressions, plus tool_selection_accuracy.

you can also just run pieces separately:

```
python train.py out/dpo_pairs.jsonl out/model Qwen/Qwen2.5-0.5B-Instruct
python eval_model.py out/model Qwen/Qwen2.5-0.5B-Instruct out/eval_pairs.jsonl out/eval_report.json out/records.jsonl
```

or try the live routing yourself:

```
python ask.py "any active stockouts?"
python ask.py "why is WH_0009 at the national_CMS failing fulfillment for vaccines?"
```

or try infering the tuned or based model yourself:

```
python try_model.py out/model "how is WH_0009 at the national_CMS doing overall?" --base Qwen/Qwen2.5-0.5B-Instruct --use_full_dataset
python try_model.py --base Qwen/Qwen2.5-0.5B-Instruct --base-only --use_full_dataset "how is WH_0009 at the national_CMS doing overall?"
```

## testing notes

ran this a bunch during dev to make sure the rubric actually distinguishes
good from broken answers across 15 different random
seeds on the real 210-row sample.

router (`assistant.route()`) gets 100% on the tool-family questions this
pipeline itself generates, across 10 seeds - makes sense since the keyword
matching was built to match the exact phrasing in `gen_pairs.py`'s
templates. real user phrasing would probably do worse; this hasn't been
tested against phrasing the router wasn't designed around.

the full pipeline that produced the final evaluation results 
(data generation through training and evaluation) was run twice, end to end, 
on the same configuration. both runs produced stable DPO training outcomes 
and evaluation results, with no meaningful run-to-run variation - reasonable
 confidence the reported numbers reflect the actual behavior of this setup, 
 not a lucky single run.
