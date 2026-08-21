"""Runs the full pipeline start to finish: load data, build prompts,
generate and label DPO preference pairs, train, and evaluate.

For the real data: place the 3 source CSVs (see CSV_FILES in
data_utils.py for the exact filenames) into data/, then run without
--sample (or point --data-dir at wherever they are).

Usage: python run.py [--sample] [--no-train] [--base-model NAME]
"""

import argparse
import json
import os
import random

import data_utils as du
import gen_pairs as G


def save_jsonl(rows, path):
    """Writes a list of dicts to a JSONL file, one JSON object per
    line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true",
                     help="use the bundled 210-row real-data sample instead of the full CSVs")
    ap.add_argument("--data-dir", default="data", help="folder with the 3 real CSVs (or the sample jsonl)")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--min-prompts", type=int, default=150)
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-new-tokens", type=int, default=G.DEFAULT_MAX_NEW,
                     help="generation length cap used during eval - same shared default as eval_model.py/try_model.py")
    ap.add_argument("--epochs", type=int, default=2,
                     help="DPO training epochs - same default train.py itself uses; exposed here so it can be swept without editing code")
    ap.add_argument("--lr", type=float, default=5e-5, help="DPO learning rate")
    ap.add_argument("--beta", type=float, default=0.3,
                     help="DPO KL regularization strength - higher keeps the tuned model closer to the reference/base model's behavior")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank - project default; higher gives the adapter more capacity but raises overfitting risk on this small a dataset")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("loading data...")
    recs = du.load_data(use_sample=args.sample, data_dir=args.data_dir)
    save_jsonl(recs, f"{args.out}/records.jsonl")
    print(f"got {len(recs)} records")

    print("building prompts...")
    prompts = G.build_prompts(recs, min_prompts=args.min_prompts)
    rng = random.Random(42)
    shuffled = prompts[:]
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * 0.15))
    eval_prompts, train_prompts = shuffled[:n_eval], shuffled[n_eval:]
    print(f"{len(prompts)} prompts, {len(train_prompts)} train / {len(eval_prompts)} held out for eval")

    print("generating candidate responses + labeling preferences...")
    train_pairs = G.gen_all(train_prompts, recs, seed=42)
    valid_cats, valid_reasons = du.valid_categories_and_reasons(recs)
    valid_levels, valid_regions = du.valid_levels_and_regions(recs)
    labeled = G.label_pairs(train_pairs, valid_cats, valid_reasons, valid_levels, valid_regions)
    save_jsonl(labeled, f"{args.out}/dpo_pairs.jsonl")

    eval_raw = G.gen_all(eval_prompts, recs, seed=999)
    eval_pairs = G.label_pairs(eval_raw, valid_cats, valid_reasons, valid_levels, valid_regions)
    save_jsonl(eval_pairs, f"{args.out}/eval_pairs.jsonl")
    print(f"wrote {len(labeled)} dpo pairs, {len(eval_pairs)} held-out eval pairs")

    if args.no_train:
        print("skipping training")
        return

    print("training with DPO...")
    import train as T
    T.train(f"{args.out}/dpo_pairs.jsonl", args.base_model, f"{args.out}/model",
            epochs=args.epochs, lr=args.lr, beta=args.beta, rank=args.rank)

    if args.no_eval:
        print("skipping eval")
        return

    print("evaluating base vs dpo model...")
    import eval_model as E
    E.run_eval(f"{args.out}/model", args.base_model, f"{args.out}/eval_pairs.jsonl",
               f"{args.out}/eval_report.json", records_file=f"{args.out}/records.jsonl",
               max_new=args.max_new_tokens)

    print("done, everything's in", args.out)


if __name__ == "__main__":
    main()
