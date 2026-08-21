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
