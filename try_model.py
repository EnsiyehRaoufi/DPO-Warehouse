"""Runs a trained model on a single live question.

Builds the prompt exactly the way training/evaluation did - route the
question, call the real tool, use the same system instructions +
"tool call: ..." context string + "Q: ...\\nA:" - so what the model sees
here matches what it saw during training.

By default loads model_path as a full model directory (works if a
merged model was saved). If only a LoRA adapter was saved, pass --base
with the original base model id and this loads it as a PEFT adapter on
top.

Usage:
    python try_model.py out/model "any active stockouts?"
    python try_model.py out/model "how is WH_0009 at the national_CMS doing overall?"
    python try_model.py out/model "why is WH_0009 at the national_CMS failing fulfillment for vaccines?" --base Qwen/Qwen2.5-0.5B-Instruct
    python try_model.py --base Qwen/Qwen2.5-0.5B-Instruct --base-only "how is WH_0009 at the national_CMS doing overall?"

Every invocation writes a timestamped log to out/try_model_log_<ts>.json
(question, routing decision, full prompt, response) so past live tests
aren't lost between runs.
"""

import argparse
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


def build_prompt(question, records):
    """Routes a question and builds the full prompt a model would
    receive, along with the routing decision for logging. Returns
    (prompt, routing_info)."""
    intent, scope = A.route(question, records)
    if intent is None:
        # no tool call to describe - the model was never trained on
        # this shape, so this is shown to make a routing failure
        # visible rather than silently feeding a misleading prompt
        context = "no matching tool/record"
        routing_info = {"intent": None, "scope": scope}
        print(f"[routing failed: {scope.get('reason', 'unknown')}]", file=sys.stderr)
    else:
        _, tool_out = A.answer(records, intent, scope)
        context = f"tool call: {intent}({scope}) -> {G.summarize_tool_out(tool_out)}"
        routing_info = {"intent": intent, "scope": scope}
        print(f"[routed to: {intent}, scope={scope}]", file=sys.stderr)

    prompt = f"{G.SYS}\n\n{context}\n\nQ: {question}\nA:"
    return prompt, routing_info


def load_model(model_path, base_model=None, base_only=False):
    """Loads either the plain base model (base_only=True, for a direct
    base-vs-tuned comparison), a base model with a LoRA adapter applied,
    or a fully merged model directory. Returns (model, tokenizer,
    device)."""
    if base_only:
        load_from = base_model
    else:
        load_from = base_model or model_path
    tok = AutoTokenizer.from_pretrained(load_from)

    if base_only:
        model = AutoModelForCausalLM.from_pretrained(base_model)
    elif base_model:
        base = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tok, device


def generate(model, tok, prompt, device, max_new=G.DEFAULT_MAX_NEW):
    """Runs the model on one prompt and returns the decoded response
    text."""
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=600).to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", nargs="?", default=None,
                     help="path to trained model dir (merged) or LoRA adapter dir - not needed with --base-only")
    ap.add_argument("question", help="your question, in quotes")
    ap.add_argument("--base", default=None,
                     help="base model id - needed if model_path is a LoRA adapter (not merged), or always with --base-only")
    ap.add_argument("--base-only", action="store_true",
                     help="skip the trained adapter entirely and run the plain base model (--base) - for a direct "
                          "base-vs-tuned comparison on the same live question")
    ap.add_argument("--max-new-tokens", type=int, default=G.DEFAULT_MAX_NEW,
                     help="generation length cap - shared default (gen_pairs.DEFAULT_MAX_NEW) with eval_model.py/run.py")
    ap.add_argument("--use_full_dataset", action="store_true",
                     help="load the full 3-CSV dataset from data/ instead of the 210-row sample")
    args = ap.parse_args()

    if args.base_only and not args.base:
        ap.error("--base-only requires --base (which model to actually run)")
    if not args.base_only and not args.model_path:
        ap.error("model_path is required unless using --base-only")

    records = du.load_data(use_sample=not args.use_full_dataset)
    prompt, routing_info = build_prompt(args.question, records)
    print(f"\n--- prompt sent to model ---\n{prompt}\n", file=sys.stderr)

    model, tok, device = load_model(args.model_path, args.base, base_only=args.base_only)
    response = generate(model, tok, prompt, device, args.max_new_tokens)

    print(f"\n--- model response ---\n{response.strip()}")

    os.makedirs("out", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "timestamp": timestamp,
        "question": args.question,
        "model_path": args.model_path,
        "base_model": args.base,
        "base_only": args.base_only,
        "routing": routing_info,
        "prompt": prompt,
        "response": response.strip(),
    }
    log_file = f"out/try_model_log_{timestamp}.json"
    json.dump(log, open(log_file, "w"), indent=2, default=str)
    print(f"\n--- log saved: {log_file} ---", file=sys.stderr)


if __name__ == "__main__":
    main()
