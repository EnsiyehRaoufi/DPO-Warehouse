"""Trains the model with DPO, using a LoRA adapter via TRL's DPOTrainer.

Default beta=0.3 keeps the tuned model closer to the reference model
than TRL's own default (0.1) - lower beta allows more drift per epoch,
which risks letting one dominant training pattern destabilize the
model's overall behavior rather than only correcting the specific thing
it was meant to teach. Default epochs=2 and rank=16 are chosen to match.
"""

import hashlib
import json
import os
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def fingerprint(path):
    """Returns a SHA-256 hash of a file's contents, used to record
    exactly which training data produced a given checkpoint."""
    h = hashlib.sha256()
    h.update(open(path, "rb").read())
    return h.hexdigest()


def train(pairs_file, base_model, out_dir, epochs=2, lr=5e-5, beta=0.3, rank=16,
          lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj")):
    """Runs DPO training on the given preference pairs file and saves
    the resulting LoRA adapter, training metrics, and run configuration
    to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(pairs_file) if l.strip()]
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
    ds = ds.train_test_split(test_size=0.1, seed=42)

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model,
                                                  torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    lora_cfg = LoraConfig(r=rank, lora_alpha=32, lora_dropout=0.05, bias="none",
                           task_type="CAUSAL_LM", target_modules=list(lora_target_modules))

    cfg = DPOConfig(
        output_dir=out_dir, beta=beta, learning_rate=lr, num_train_epochs=epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        # eval batch size explicitly set to 1 (not left at HF's default
        # of 8) since DPO evaluation needs an extra reference-model
        # forward pass plus fp32 logit upcasting that training steps
        # don't do, which can exhaust GPU memory at a larger eval batch
        # size. eval_accumulation_steps=1 moves eval outputs to CPU
        # after each batch instead of holding them all on GPU until the
        # end, for the same reason.
        per_device_eval_batch_size=1, eval_accumulation_steps=1,
        max_length=768, truncation_mode="keep_start",
        use_cpu=not torch.cuda.is_available(),
        logging_steps=10, eval_strategy="steps", eval_steps=50, save_strategy="epoch",
        report_to=[], remove_unused_columns=False,
    )

    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["test"],
                          processing_class=tok, peft_config=lora_cfg)
    result = trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    # metrics plus the exact configuration that produced this
    # checkpoint, so any later evaluation can report what was actually
    # used without needing to remember or dig through console logs
    json.dump({"log_history": trainer.state.log_history, "final": result.metrics},
               open(f"{out_dir}/metrics.json", "w"), indent=2, default=str)
    json.dump({
        "base_model": base_model, "epochs": epochs, "lr": lr, "beta": beta,
        "lora_rank": rank, "lora_alpha": 32, "lora_dropout": 0.05,
        "lora_target_modules": list(lora_target_modules),
        "pairs_file": pairs_file, "dataset_fingerprint": fingerprint(pairs_file),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, open(f"{out_dir}/run_info.json", "w"), indent=2)

    return trainer


if __name__ == "__main__":
    # usage: python train.py [pairs_file] [out_dir] [base_model] [epochs] [lr] [beta] [rank]
    pairs_file = sys.argv[1] if len(sys.argv) > 1 else "out/dpo_pairs.jsonl"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "out/model"
    base_model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-0.5B-Instruct"
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 5e-5
    beta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
    rank = int(sys.argv[7]) if len(sys.argv) > 7 else 16
    train(pairs_file, base_model, out_dir, epochs=epochs, lr=lr, beta=beta, rank=rank)
