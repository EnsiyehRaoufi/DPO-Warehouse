# feedback rubric

what the scoring in `gen_pairs.py score()` checks (same 7 things a human
reviewer should check if they're filling in real labels instead) - note:
tool/function selection ("did it call kpis vs stockouts vs explain_issue")
is NOT one of these 7. it's measured separately (see "tool-use correctness
is not a score() dimension" below), because it's a rule-based decision
that happens before either model ever runs, not something the LLM itself
does:

1. **grounding** - do the numbers/ids/values in the response actually match
   the data?
2. **operational_quality** - is the answer well-formed and finished, not a 
   rambling wall of text? scored on three structural checks, not raw length: 
   (a) states its FACTS/RECOMMENDATION structure once each, not looping back 
   and re-stating itself under a fresh label; 
   (b) doesn't repeat the same number 3+ times; 
   (c) ends on a
   finished sentence, not cut off mid-word. 
   each failure is a 1/3 penalty off 1.0. 
3. **uncertainty** - despite the name, this does NOT measure how much
   uncertainty the model expressed - it measures whether the amount it
   expressed was CORRECT. does it say "not sure" (or "check the
   system"/"check yourself") when the data really is missing/
   provisional, AND does it NOT say those things when the data is fine?
   a confident, hedge-free answer on a fully-answerable question scores
   1.0 here; a hedge on that same answerable question scores 0.0 -
   hedging isn't rewarded for its own sake. 
4. **no hallucination** - doesn't make up a warehouse id that isn't real.
5. **no invented category** - doesn't state a commodity category that isn't
   one of the real ones in the data.
6. **no invented reason** - doesn't state an inventory-issue reason code
   that isn't one of the real 12 in the data.
7. **no invented group** - for a compare-type answer, doesn't state a
   warehouse tier or region type that isn't one of the real ones in the
   data (e.g. "provincial_store" or "suburban").

if a response invents a warehouse id, category, reason code, or tier/
region label, that should basically always lose, doesn't matter how good
it is otherwise.

## tool-use correctness is not a `score()` dimension

`tool_use` correctness - "did `assistant.route()` pick the right one of
the 5 tools, with the right scope, from the raw question text" - is
measured separately, by `evaluate_tool_selection()` in `eval_model.py`.
that's a **rule-based keyword classifier**, not something either model
does: it runs on the question text alone, before base or DPO ever
generates anything, so its accuracy is identical no matter which model you
evaluate. `eval_report.json` reports it as
`tool_use_correctness_before`/`tool_use_correctness_after` with the SAME
value in both fields (plus a `tool_use_correctness_note` explaining why),
rather than as its own separate field, to satisfy the assignment's
explicit "tool-use correctness" before/after comparison requirement while
staying honest that this specific number cannot move between the two.

## filter vs. compare vs. single-warehouse - the three ways a question can be scoped

all 5 tools - `get_stockouts`/`get_backorders`/`rank_shortage_risk`
(dataset-wide by nature) and now `get_kpis`/`get_dominant_issue` too
(generalized from the single-warehouse-only `get_warehouse_kpis`/
`explain_inventory_issue`) - can each be asked the same 5 ways, resolved
by `assistant.resolve_dataset_wide_scope()`:

| question | scope shape | example |
|---|---|---|
| unscoped | `{}` | "which items are backordered?" |
| filter (1 tier/region named) | `{"warehouse_level": "district_store"}` | "...in the district tier?" |
| compare all | `{"compare_by": "warehouse_level", "compare_values": [all 3]}` | "compare backorders across tiers" |
| compare named (2+ named) | `{"compare_by": "warehouse_level", "compare_values": [2 named]}` | "how do district and national compare?" |
| single warehouse named | `{"uid": "national_CMS:WH_0009"}` | "which category has the most backorders at WH_0009?" |

a specific named warehouse takes full priority over tier/region language
in the same sentence - a single warehouse already implies one fixed tier
and one fixed region, so filtering further on top of it would be
redundant at best, contradictory at worst. if the named id is ambiguous
(exists at multiple tiers, no tier specified) or unknown, routing fails
cleanly with a reason (matching how `explain_issue`'s own single-warehouse
branch already handles this exact ambiguity) instead of silently falling
back to an unscoped or wrong-scoped answer.

**`kpis` and `explain_issue` get the same 5 shapes, with one semantic
difference worth being explicit about.** `get_kpis(records, **filters)`
generalizes `get_warehouse_kpis` directly - same averaging math
(`_compute_kpis()`), just over a wider or narrower row-set, so "kpis for
the national tier" is a straightforward real aggregate, no different in
kind from "backorders in the national tier." `explain_issue` is
different: a single warehouse's answer is grounded in ONE real record's
actual recorded reason - there's no equivalent single "the reason" a
tier or region can point to, since different warehouses in the same
tier fail for different real reasons. so the tier/region/compare version
of `explain_issue` (`get_dominant_issue()`) answers a related but
distinct question: the most common REAL issue code among matching
records, grounded in an actual count - not a guess, not an average, not
a synthesized explanation. "why is the national tier failing" gets "the
most common recorded issue is X, seen N times," not a fabricated causal
narrative.

## the 7 dims above vs. eval_report.json's field names - these do NOT match 1:1

this was a real source of confusion, not just missing docs: `eval_model.py`
renames one dimension and sources `tool_use_correctness` from an entirely
separate mechanism (not from `score()` at all - see "tool-use correctness
is not a `score()` dimension" above), and `eval_report.json` didn't
explain itself (fixed - it now carries a `field_descriptions` block, and
every entry in `top_improvements` / `top_regressions` / `regression_cases`
carries the full 7-dim breakdown for both models under
`dims_base`/`dims_dpo`, not just the combined total score). the mapping:

| report field | = | built from |
|---|---|---|
| `factual_correctness_*` | = | `avg(grounding)` |
| `tool_use_correctness_*` | = | NOT from `score()` - `evaluate_tool_selection()`'s router accuracy, identical value in `_before` and `_after` by construction |
| `operational_quality_*` | = | `avg(operational_quality)` - one rubric dimension directly (formerly a composite of `clarity` + `useful`; `useful` was folded into `uncertainty` instead - see above) |
| `uncertainty_handling_*` | = | `avg(uncertainty)` - now also covers what `useful` used to check |
| `hallucination_rate_*` | = | fraction of responses where `no_hallucination < 1` OR `no_invented_category < 1` OR `no_invented_reason < 1` OR `no_invented_group < 1` - fabrication only, does NOT include `grounding < 0.5` (that would conflate incompleteness with fabrication - removed) |
| `baseline_score` / `post_dpo_score` | = | a WEIGHTED average of all 7 dims - NOT equal 1/7 each. `operational_quality` is weighted 10%, the other 6 split the remaining 90% equally (15% each). see `eval_model.py`'s `_EVAL_WEIGHTS`/`weighted_total()` for the exact numbers and the reasoning - `gen_pairs.py`'s `score()` (used for DPO training-pair labeling) stays equal-weighted; this reweighting is eval-reporting only |

## `win_rate_vs_baseline` - not built from a single dimension either

different kind of field than the table above - it's not an average of one
rubric dimension, it's a **per-prompt head-to-head comparison** of the two
models' TOTAL scores (all 7 dims combined):

```
delta_per_prompt = dpo_total_score - base_total_score   (computed per eval prompt)
win_rate_vs_baseline = count(delta_per_prompt > 0) / n_eval_prompts
```

a prompt counts as a win only if DPO's total score is STRICTLY higher than
base's on that exact question - ties (`delta == 0`) and losses
(`delta < 0`) both count against it, only strict wins count for it. this
makes it a stricter, more informative signal than `delta` (the plain
average): a positive `delta` can happen even if DPO only won a minority of
individual prompts, as long as the wins were large enough to outweigh the
losses/ties in the average. `n_improved`/`n_tied`/`n_regressions` in the
report break out the raw counts behind this fraction (`win_rate_vs_baseline
== n_improved / n_eval`), and `run_eval()` prints this breakdown directly
to the console at the end of every run, not just to the json file.

## grounding tolerance fix - found via real DPO eval regression analysis

after a real DPO training run, 13 of 14 regression cases (the trained
model scoring lower than base on a held-out prompt) traced back almost
entirely to the `grounding` dimension. inspecting the actual response
text showed roughly half of those weren't real errors at all - the
DPO-tuned model stated the exact same correct facts, just phrased more
naturally than the raw stored value: `"national_CMS"` (raw) vs `"National
CMS"` (natural phrasing), `"82.0"` (raw) vs `"82%"` (trailing zero
dropped). the old exact-substring check had zero tolerance for this and
penalized it as if the value were wrong.

fixed: numeric comparison now allows a small tolerance (0.05) instead of
requiring an exact string match, and string comparison normalizes case
and underscore-vs-space before comparing. 

## faithfulness vs hallucination - what this actually measures

worth being precise about this: dimensions 1, 4, 5, 6, 7 are closer to a
RAG **faithfulness** check (does the response stay consistent with the
tool output it was given) than general-purpose hallucination detection.
4/5/6/7 specifically catch *fabricated entities* (an id/category/reason/
tier/region that doesn't exist anywhere in the data) - that's real
hallucination detection, but only for those 4 fields, using structural
regex extraction tied to our own fixed response phrasing. 
