"""Zero-shot comparison of candidate base models, before spending time
on DPO training any of them. Doesn't train anything - runs each
candidate on the same eval prompts, scores them with the same rubric
eval_model.py uses for before/after, and ranks them by average score.

This is how a base model choice can actually be checked against
alternatives, rather than picked as a reasonable-sounding default.

Usage:
    python compare_models.py --sample --n-prompts 20
    python compare_models.py --candidates "Qwen/Qwen2.5-0.5B-Instruct,HuggingFaceTB/SmolLM2-360M-Instruct"
"""

import argparse
import gc
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import data_utils as du
import eval_model as E
import gen_pairs as G

DEFAULT_CANDIDATES = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


def load_eval_prompts(use_sample, data_dir, n_prompts, seed):
    """Builds a small set of scored prompts to run every candidate
    model against."""
    recs = du.load_data(use_sample=use_sample, data_dir=data_dir)
    prompts = G.build_prompts(recs, min_prompts=n_prompts)[:n_prompts]
    raw = G.gen_all(prompts, recs, seed=seed)
    valid_cats, valid_reasons = du.valid_categories_and_reasons(recs)
    valid_levels, valid_regions = du.valid_levels_and_regions(recs)
    pairs = G.label_pairs(raw, valid_cats, valid_reasons, valid_levels, valid_regions)
    return pairs, valid_cats, valid_reasons, valid_levels, valid_regions


def evaluate_one_model(model_name, eval_pairs, device, valid_categories=None, valid_reasons=None,
                        valid_levels=None, valid_regions=None):
    """Loads one candidate model, runs it over every eval prompt, and
    returns its summary score plus per-prompt results. Frees the model
    from memory before returning, since these can add up quickly across
    several candidates."""
    print(f"\n=== {model_name} ===")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    results = E.run_model(model, tok, eval_pairs, device, valid_categories, valid_reasons,
                           valid_levels, valid_regions)
    summary = E.summarize(results)

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, results


def compare(candidates, eval_pairs, out_file, n_sample_responses=2, valid_categories=None, valid_reasons=None,
            valid_levels=None, valid_regions=None):
    """Runs every candidate model over the same eval prompts, ranks them
    by average score, and writes a comparison report."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report = {"n_eval_prompts": len(eval_pairs), "candidates": {}}
    for name in candidates:
        try:
            summary, results = evaluate_one_model(name, eval_pairs, device, valid_categories, valid_reasons,
                                                    valid_levels, valid_regions)
        except Exception as e:
            print(f"  FAILED to evaluate {name}: {e}")
            report["candidates"][name] = {"error": str(e)}
            continue
        summary["sample_responses"] = [
            {"qid": r["qid"], "response": r["response"]} for r in results[:n_sample_responses]
        ]
        report["candidates"][name] = summary
        print({k: v for k, v in summary.items() if k != "sample_responses"})

    ranked = sorted(
        [(name, r["avg_score"]) for name, r in report["candidates"].items() if "avg_score" in r],
        key=lambda x: x[1], reverse=True,
    )
    report["ranking"] = ranked

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== ranking (best zero-shot avg_score first) ===")
    for name, score in ranked:
        print(f"  {score:.3f}  {name}")
    failed = [n for n in candidates if "error" in report["candidates"].get(n, {})]
    if failed:
        print("  (failed to load/run:", ", ".join(failed), ")")
    if ranked:
        print(f"\nbest candidate: {ranked[0][0]}")
        print("(this is ZERO-SHOT performance, before any DPO training - it tells you")
        print(" which model is a good starting point, not how well DPO will do on it)")

    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES),
                     help="comma-separated HF model names to compare")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-prompts", type=int, default=20,
                     help="keep this small - generation runs once per prompt per candidate model")
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--out", default="out/model_comparison.json")
    args = ap.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    eval_pairs, valid_cats, valid_reasons, valid_levels, valid_regions = load_eval_prompts(
        args.sample, args.data_dir, args.n_prompts, args.seed)
    print(f"comparing {len(candidates)} candidate models on {len(eval_pairs)} eval prompts (zero-shot, no training)...")
    compare(candidates, eval_pairs, args.out, valid_categories=valid_cats, valid_reasons=valid_reasons,
            valid_levels=valid_levels, valid_regions=valid_regions)
