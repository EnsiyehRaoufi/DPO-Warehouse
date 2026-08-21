# assets

Supporting documentation and evaluation artifacts for the project -
everything here explains or backs up what's in the main pipeline
(the `.py` files and `README.md`/`RUBRIC.md` at the repo root), rather
than being part of the pipeline itself.

## Files

- **`ARCHITECTURE.md`** - how the pipeline fits together: the offline
  flow (data → DPO training pairs → training → evaluation) and the live
  query flow (a question in, a grounded answer out), each as a diagram
  plus a plain-language walkthrough.
- **`PROJECT_EVALUATION_REPORT.md`** - project history, base model
  selection (with the comparison numbers behind that choice), the
  training environment, and the final evaluation results with charts.
- **`eval_report_20260821_085504.json`** - the evaluation report from
  the training run `PROJECT_EVALUATION_REPORT.md` reports on: summary
  scores, before/after deltas, and the top improvements/regressions.
- **`eval_report_log_20260821_085504.json`** - the companion log for
  that same run: every response either model generated (not just the
  handful shown in the report), plus the full regression list.
- **`visualize_eval_report.py`** - the script that turns an eval report
  (and optionally its log) into the PNG charts in `img/`. Also accepts
  multiple report files at once to plot trends across several training
  runs, not just one.
- **`img/`** - the PNG charts `PROJECT_EVALUATION_REPORT.md` embeds:
  before/after scores per dimension, the win/tie/regress split, the
  aggregate score, and what's driving the regressions.

## Regenerating the plots

```
python assets/visualize_eval_report.py assets/eval_report_20260821_085504.json \
    --log assets/eval_report_log_20260821_085504.json \
    --out-dir assets/img
```



# assets

Supporting documentation and evaluation artifacts for the project -
everything here explains or backs up what's in the main pipeline
(the `.py` files and `README.md`/`RUBRIC.md` at the repo root), rather
than being part of the pipeline itself.

## Files

- **`ARCHITECTURE.md`** - how the pipeline fits together: the offline
  flow (data → DPO training pairs → training → evaluation) and the live
  query flow (a question in, a grounded answer out), each as a diagram
  plus a plain-language walkthrough.
- **`PROJECT_EVALUATION_REPORT.md`** - project history, base model
  selection (with the comparison numbers behind that choice), the
  training environment, and the final evaluation results with charts.
- **`eval_report_20260821_085504.json`** - the evaluation report from
  the training run `PROJECT_EVALUATION_REPORT.md` reports on: summary
  scores, before/after deltas, and the top improvements/regressions.
- **`eval_report_log_20260821_085504.json`** - the companion log for
  that same run: every response either model generated (not just the
  handful shown in the report), plus the full regression list.
- **`visualize_eval_report.py`** - the script that turns an eval report
  (and optionally its log) into the PNG charts in `img/`. Also accepts
  multiple report files at once to plot trends across several training
  runs, not just one.
- **`img/`** - the PNG charts `PROJECT_EVALUATION_REPORT.md` embeds:
  before/after scores per dimension, the win/tie/regress split, the
  aggregate score, and what's driving the regressions.

## Preview: "Final Evaluation Results" from `PROJECT_EVALUATION_REPORT.md`

An excerpt - see the
[full report](https://github.com/EnsiyehRaoufi/DPO-Warehouse/blob/main/assets/PROJECT_EVALUATION_REPORT.md#final-evaluation-results)
for the charts and complete regression analysis:

> **Summary, on 60 held-out evaluation prompts:**
>
> | Metric | Value |
> |---|---|
> | Baseline score | 0.8869 |
> | Post-DPO score | 0.9002 |
> | Delta | **+0.0133** |
> | Win rate vs. baseline | 46.7% |
> | Improved | 28 |
> | Tied | 17 |
> | Regressed | 15 |
>
> - **Factual correctness (grounding)** held steady (0.616 → 0.616): the
>   tuned model preserves its ability to state real, correct facts from
>   the data at the same rate as the base model.
> - **Operational quality** improved (0.694 → 0.778): responses are more
>   consistently well-structured, non-redundant, and complete after
>   tuning.
> - **Hallucination rate** improved (5.0% → 3.3%): fewer responses
>   invent a warehouse ID, category, reason code, or tier/region label
>   that doesn't exist in the data.
> - **Uncertainty handling** improved (0.883 → 0.900) - and, notably,
>   none of the 15 remaining regressions in this run involve unwarranted
>   hedging at all.

## Regenerating the plots

```
python assets/visualize_eval_report.py assets/eval_report_20260821_085504.json \
    --log assets/eval_report_log_20260821_085504.json \
    --out-dir assets/img
```
