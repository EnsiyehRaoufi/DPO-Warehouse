"""Runs evaluation before and after DPO training - same prompts, same
scoring, for both the base model and the tuned model - and writes a
timestamped report plus a separate detailed log file.
"""

import datetime
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import assistant as A
import data_utils as du
import gen_pairs as G

# operational_quality is weighted down to 10% here, for evaluation
# reporting only - gen_pairs.py's score() itself stays equal-weighted
# (1/7 each), since that's what decides chosen/rejected during DPO
# training-pair labeling, and this reweighting is scoped to evaluation
# only. The other 6 dimensions split the remaining 90% equally (15%
# each), so relative to each other they're unchanged - only
# operational_quality's own share shrank.
#
# why: across several training runs, operational_quality reliably
# improved after DPO while grounding/uncertainty sometimes still got
# worse in those same runs. under equal weighting, that reliable
# structural improvement could offset - and hide - a real regression
# in whether the content was actually correct, in the single headline
# eval number. operational_quality is still a real quality signal
# worth training against; this only changes how much it counts toward
# the aggregate eval score, not whether it's tracked
# (operational_quality_before/_after are still reported in full below)
# or trained against.
_EVAL_WEIGHTS = {
    "operational_quality": 0.10,
    "grounding": 0.90 / 6,
    "uncertainty": 0.90 / 6,
    "no_hallucination": 0.90 / 6,
    "no_invented_category": 0.90 / 6,
    "no_invented_reason": 0.90 / 6,
    "no_invented_group": 0.90 / 6,
}


def weighted_total(dims):
    """Combines a per-dimension score breakdown into one aggregate
    score using _EVAL_WEIGHTS, for evaluation reporting."""
    return sum(dims[k] * w for k, w in _EVAL_WEIGHTS.items())


def load_training_settings(model_path):
    """Reads run_info.json (written by train.py) from the trained model
    directory, if it exists. Returns None if not found - e.g. a
    --base-only comparison with no trained adapter, or a checkpoint
    saved before run_info.json existed."""
    path = os.path.join(model_path, "run_info.json") if model_path else None
    if path and os.path.exists(path):
        return json.load(open(path))
    return None


def gen_response(model, tok, prompt, device, max_new=G.DEFAULT_MAX_NEW):
    """Generates one response from a model for one prompt."""
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=600).to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def run_model(model, tok, eval_pairs, device, valid_categories=None, valid_reasons=None,
              valid_levels=None, valid_regions=None, max_new=G.DEFAULT_MAX_NEW):
    """Runs a model over every eval prompt and scores each response.
    Returns a list of {qid, response, score, dims}."""
    results = []
    for p in eval_pairs:
        text = gen_response(model, tok, p["prompt"], device, max_new=max_new)
        _, dims = G.score(text, p["tool_out"], p["scope"], p["uncertain"],
                           valid_categories, valid_reasons, valid_levels, valid_regions)
        # score() itself returns an equal-weighted total (used elsewhere
        # for training-pair labeling) - eval reporting uses the
        # separately-weighted total instead, see _EVAL_WEIGHTS above
        results.append({"qid": p["qid"], "response": text, "score": weighted_total(dims), "dims": dims})
    return results


def evaluate_tool_selection(eval_pairs, records):
    """Measures routing accuracy: how often assistant.route() (a
    rule-based keyword classifier) picks the correct tool and scope from
    raw question text. This is the same value regardless of which model
    (base or DPO) is passed in elsewhere, since routing happens before
    either model runs and neither model's weights can affect it. Only
    "tool" family prompts are checked (the "field" family ones aren't
    routed through the 5 required tools)."""
    tool_pairs = [p for p in eval_pairs if p.get("family") == "tool"]
    if not tool_pairs:
        return {"n": 0, "accuracy": None}

    correct = 0
    misses = []
    for p in tool_pairs:
        raw_question = p["prompt"].split("Q: ")[-1].split("\nA:")[0]
        pred_intent, pred_scope = A.route(raw_question, records)
        true_scope = p["scope"]
        ok = pred_intent == p["intent"]
        # checks every scope key the true prompt actually has, not just
        # uid, so a filter/compare prompt with a wrong or missing
        # warehouse_level/region_type/compare_by doesn't count as
        # correct just because the intent matched
        for key in ("uid", "warehouse_level", "region_type", "compare_by"):
            if key in true_scope:
                ok = ok and pred_scope.get(key) == true_scope[key]
        if "compare_values" in true_scope:
            ok = ok and sorted(pred_scope.get("compare_values") or []) == sorted(true_scope["compare_values"])
        correct += int(ok)
        if not ok:
            misses.append({"qid": p["qid"], "question": raw_question,
                            "true_intent": p["intent"], "predicted_intent": pred_intent,
                            "true_scope": true_scope, "predicted_scope": pred_scope})
    return {"n": len(tool_pairs), "accuracy": correct / len(tool_pairs), "misses": misses}


def summarize(results):
    """Aggregates a list of per-response results into summary
    statistics: average score per dimension, plus a combined
    hallucination_rate covering any of the 4 fabrication-related
    dimensions (invented warehouse id, category, reason code, or
    tier/region label)."""
    n = max(1, len(results))
    avg = lambda k: sum(r["dims"][k] for r in results) / n
    halluc = sum(
        1 for r in results
        if r["dims"]["no_hallucination"] < 1
        or r["dims"].get("no_invented_category", 1.0) < 1
        or r["dims"].get("no_invented_reason", 1.0) < 1
        or r["dims"].get("no_invented_group", 1.0) < 1
    )
    return {
        "avg_score": sum(r["score"] for r in results) / n,
        "factual_correctness": avg("grounding"),
        "operational_quality": avg("operational_quality"),
        "uncertainty_handling": avg("uncertainty"),
        "hallucination_rate": halluc / n,
    }


# every report field explained in plain language, with the exact
# formula for anything that isn't a raw rubric dimension
FIELD_DESCRIPTIONS = {
    "training_settings": "the DPO training config that produced the model being evaluated (LoRA rank, beta, epochs, learning rate, base model, dataset fingerprint) - read from run_info.json in the model directory, written by train.py. null if no run_info.json was found (e.g. a --base-only comparison, or an older checkpoint saved before run_info.json existed)",
    "n_eval": "number of held-out eval prompts scored (never seen during DPO training)",
    "baseline_score": "avg total rubric score (0-1) of the un-tuned base model's responses, per response then across all prompts - NOT an equal 1/7-per-dimension average: operational_quality is weighted 10%, the other 6 dimensions split the remaining 90% equally (15% each) - see _EVAL_WEIGHTS/weighted_total() in this file for why. gen_pairs.py's score() itself stays equal-weighted; this reweighting is eval-reporting only, doesn't affect DPO training-pair labeling",
    "post_dpo_score": "same computation as baseline_score, but for the DPO-tuned model",
    "delta": "post_dpo_score - baseline_score (positive = DPO improved the average)",
    "win_rate_vs_baseline": "fraction of prompts where DPO's response scored STRICTLY higher than base's on that same prompt (stricter than delta - a positive delta can still have a win rate well under 50% if most prompts tied) = n_improved / n_eval",
    "n_improved": "count of prompts where the DPO model scored STRICTLY higher than the base model (delta > 0)",
    "n_tied": "count of prompts where DPO and base scored exactly the same (delta == 0)",
    "hallucination_rate_before / _after": "fraction of base/DPO responses flagged as fabricating something that doesn't exist: an invented warehouse id (no_hallucination dim), an invented commodity category (no_invented_category dim), an invented issue reason code (no_invented_reason dim), or an invented tier/region label in a compare answer (no_invented_group dim). does NOT include incomplete/wrong-but-real facts - that's factual_correctness's job, not this one",
    "factual_correctness_before / _after": "= avg(grounding) - fraction of the real tool-output values (numbers, ids) that actually appear correctly in the response text",
    "tool_use_correctness_before / _after": "accuracy of assistant.route() (a rule-based keyword classifier) at picking the correct one of the 5 tools + correct scope from raw question text. NOT a rubric dimension from score() - computed once via evaluate_tool_selection() and reported under both _before and _after with the SAME value, because routing happens before either model ever runs and neither model's weights can affect it; a _before/_after split would misleadingly imply DPO could move this number. see tool_use_correctness_note and evaluate_tool_selection()'s docstring for why",
    "tool_use_correctness_note": "explains why tool_use_correctness_before == tool_use_correctness_after by construction (see above)",
    "operational_quality_before / _after": "= avg(operational_quality) - 1.0 minus 1/3 per structural failure: looping FACTS:/Fact:/RECOMMENDATION: labels, the same number repeated 3+ times, or an unfinished/truncated sentence. NOT a word count. formerly named 'clarity' and formerly a composite with a separate 'useful' dimension - useful's checks are now folded into uncertainty_handling instead (see that entry), so this is a single rubric dimension reported directly, not a composite of two",
    "uncertainty_handling_before / _after": "= avg(uncertainty) - discloses uncertainty when the data genuinely is incomplete/provisional, and does NOT hedge when the data is solid. also covers what a separate 'useful' dimension used to check (hedge phrases like 'check the system'/'check yourself') - folded in here so hedging is judged once, against whether it was actually warranted, instead of being penalized unconditionally by one dimension while being correctly rewarded by another",
    "n_regressions": "count of prompts where the DPO model scored LOWER than the base model",
    "top_improvements / top_regressions": "the top_k prompts with the largest positive/negative score delta, including each response's full per-dimension score breakdown (dims_base/dims_dpo) so you can see exactly which of the 7 dimensions changed and by how much, not just the total",
}


def run_eval(model_path, base_model, eval_pairs_file, out_file, top_k=5, records_file=None,
             max_new=G.DEFAULT_MAX_NEW):
    """Runs the full before/after evaluation: loads the base model and
    the DPO-tuned model, scores both against the same held-out prompts,
    and writes a timestamped report and a separate detailed log file.
    Returns the report dict."""
    training_settings = load_training_settings(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_pairs = [json.loads(l) for l in open(eval_pairs_file) if l.strip()]

    # loaded early so the fabricated-category/reason/tier/region checks
    # in score() can run too, not just the warehouse-id hallucination
    # check
    valid_cats, valid_reasons, valid_levels, valid_regions = None, None, None, None
    records = None
    if records_file:
        records = [json.loads(l) for l in open(records_file) if l.strip()]
        valid_cats, valid_reasons = du.valid_categories_and_reasons(records)
        valid_levels, valid_regions = du.valid_levels_and_regions(records)

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("running base model...")
    base = AutoModelForCausalLM.from_pretrained(base_model).to(device).eval()
    base_results = run_model(base, tok, eval_pairs, device, valid_cats, valid_reasons, valid_levels, valid_regions,
                              max_new=max_new)
    base_summary = summarize(base_results)

    print("running dpo model...")
    tuned = PeftModel.from_pretrained(base, model_path).to(device).eval()
    dpo_results = run_model(tuned, tok, eval_pairs, device, valid_cats, valid_reasons, valid_levels, valid_regions,
                             max_new=max_new)
    dpo_summary = summarize(dpo_results)

    diffs = []
    for b, d in zip(base_results, dpo_results):
        diffs.append({
            "qid": b["qid"], "base_score": b["score"], "dpo_score": d["score"],
            "delta": d["score"] - b["score"], "base_resp": b["response"], "dpo_resp": d["response"],
            "dims_base": b["dims"], "dims_dpo": d["dims"],
        })
    diffs.sort(key=lambda x: x["delta"], reverse=True)
    top_improvements = diffs[:top_k]
    top_regressions = sorted(diffs, key=lambda x: x["delta"])[:top_k]
    regressions = [d for d in diffs if d["delta"] < 0]
    n_improved = sum(1 for d in diffs if d["delta"] > 0)
    n_tied = sum(1 for d in diffs if d["delta"] == 0)
    n_regressed = len(regressions)
    win_rate = n_improved / max(1, len(diffs))

    print(f"\nvs baseline, on the {len(diffs)} held-out eval prompts:")
    print(f"  {n_improved} improved, {n_tied} tied, {n_regressed} regressed "
          f"(win_rate_vs_baseline = {win_rate:.3f})")

    # computed here, before building the report, so both _before and
    # _after can be set to the same value: routing runs on the question
    # text alone before either model generates anything, so neither
    # model's weights can change this number. 
    if records_file:
        tool_sel = evaluate_tool_selection(eval_pairs, records)
        tool_use_correctness = tool_sel["accuracy"]
        tool_use_note = ("identical before/after by construction - tool/function selection in this pipeline is a "
                          "rule-based keyword router (assistant.route()), not something either model performs, so "
                          "it cannot differ between base and DPO. see evaluate_tool_selection()'s docstring in "
                          "this file for details.")
    else:
        tool_sel = None
        tool_use_correctness = None
        tool_use_note = "not computed - no records_file provided"

    report = {
        "training_settings": training_settings,
        "field_descriptions": FIELD_DESCRIPTIONS,
        "n_eval": len(eval_pairs),
        "baseline_score": base_summary["avg_score"],
        "post_dpo_score": dpo_summary["avg_score"],
        "delta": dpo_summary["avg_score"] - base_summary["avg_score"],
        "win_rate_vs_baseline": win_rate,
        "n_improved": n_improved,
        "n_tied": n_tied,
        "factual_correctness_before": base_summary["factual_correctness"],
        "factual_correctness_after": dpo_summary["factual_correctness"],
        "tool_use_correctness_before": tool_use_correctness,
        "tool_use_correctness_after": tool_use_correctness,
        "tool_use_correctness_note": tool_use_note,
        "operational_quality_before": base_summary["operational_quality"],
        "operational_quality_after": dpo_summary["operational_quality"],
        "hallucination_rate_before": base_summary["hallucination_rate"],
        "hallucination_rate_after": dpo_summary["hallucination_rate"],
        "uncertainty_handling_before": base_summary["uncertainty_handling"],
        "uncertainty_handling_after": dpo_summary["uncertainty_handling"],
        "n_regressions": n_regressed,
        "top_improvements": top_improvements,
        "top_regressions": top_regressions,
    }

    if tool_sel is not None:
        report["tool_selection_n"] = tool_sel["n"]
        report["tool_selection_misses"] = tool_sel.get("misses", [])

    # training_settings is repeated here too so the log is
    # self-contained.
    log = {
        "training_settings": training_settings,
        "base_responses": base_results,
        "dpo_responses": dpo_results,
        "regression_cases": regressions,
    }

    # both files are timestamped so re-running doesn't overwrite a
    # previous run's results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root, ext = os.path.splitext(out_file)
    timestamped_report_file = f"{root}_{timestamp}{ext}"
    log_file = f"{root}_log_{timestamp}{ext}"

    json.dump(report, open(timestamped_report_file, "w"), indent=2)
    json.dump(log, open(log_file, "w"), indent=2, default=str)
    print(f"report: {timestamped_report_file}")
    print(f"log: {log_file}")
    print({k: v for k, v in report.items()
           if not isinstance(v, list) and k != "field_descriptions"})
    return report


if __name__ == "__main__":
    # usage: python eval_model.py [model_path] [base_model] [eval_file] [out_file] [records_file] [max_new_tokens]
    model_path = sys.argv[1] if len(sys.argv) > 1 else "out/model"
    base_model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B-Instruct"
    eval_file = sys.argv[3] if len(sys.argv) > 3 else "out/eval_pairs.jsonl"
    out_file = sys.argv[4] if len(sys.argv) > 4 else "out/eval_report.json"
    records_file = sys.argv[5] if len(sys.argv) > 5 else "out/records.jsonl"
    max_new = int(sys.argv[6]) if len(sys.argv) > 6 else G.DEFAULT_MAX_NEW
    run_eval(model_path, base_model, eval_file, out_file, records_file=records_file, max_new=max_new)
