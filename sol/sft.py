"""LoRA SFT of Qwen3.6-27B on generated entries (one condition per invocation).

Each sample: system + user(prompt) + assistant(no-think prefill + response).
Loss on the assistant span only (prefill included, so students also learn the
no-think format). Manual training loop — fewer version quirks than Trainer.

Usage: python sft.py --condition C6 [--epochs 3] [--push]
Saves adapter to loras/<cond> and optionally pushes to lukebaines (private).
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SYSTEM, NOTHINK_PREFILL

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MAX_LEN = 1024


def build_sample(tok, prompt, response):
    prefix = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\n{prompt}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    target = NOTHINK_PREFILL + response + "<|im_end|>"
    pre_ids = tok.encode(prefix, add_special_tokens=False)
    tgt_ids = tok.encode(target, add_special_tokens=False)
    ids = (pre_ids + tgt_ids)[:MAX_LEN]
    labels = ([-100] * len(pre_ids) + tgt_ids)[:MAX_LEN]
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["C6", "C6e"])
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--mbs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in
            open(os.path.join(RES, f"train_{args.condition.lower()}.jsonl"))]
    data = [build_sample(tok, r["prompt"], r["response"]) for r in rows]
    print(f"{len(data)} samples, median len "
          f"{sorted(len(x[0]) for x in data)[len(data)//2]} tokens")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda")
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed for checkpointing + frozen base

    names = {n.split(".")[-1] for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear)}
    targets = sorted(names & {"q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"})
    print("LoRA targets:", targets)
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                      target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    rng = random.Random(0)
    steps_per_epoch = math.ceil(len(data) / (args.mbs * args.accum))
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, total_steps)
    pad = tok.pad_token_id or 0

    model.train()
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for s in range(steps_per_epoch):
            batch_idx = order[s * args.mbs * args.accum:(s + 1) * args.mbs * args.accum]
            opt.zero_grad(set_to_none=True)
            tot_loss = 0.0
            for a in range(0, len(batch_idx), args.mbs):
                chunk = [data[i] for i in batch_idx[a:a + args.mbs]]
                L = max(len(x[0]) for x in chunk)
                ids = torch.tensor([x[0] + [pad] * (L - len(x[0])) for x in chunk],
                                   device="cuda")
                labels = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in chunk],
                                      device="cuda")
                attn = (ids != pad).long()
                out = model(input_ids=ids, attention_mask=attn, labels=labels)
                loss = out.loss / max(1, math.ceil(len(batch_idx) / args.mbs))
                loss.backward()
                tot_loss += float(loss)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0 or step == total_steps:
                print(f"epoch {ep} step {step}/{total_steps} loss {tot_loss:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    outdir = os.path.join(HERE, "loras", args.condition.lower())
    model.save_pretrained(outdir)
    print(f"adapter saved to {outdir}")
    if args.push:
        from huggingface_hub import HfApi
        repo = f"lukebaines/gcst-qwen3.6-27b-lora-{args.condition.lower()}"
        api = HfApi()
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=outdir, repo_id=repo)
        print(f"pushed to https://huggingface.co/{repo} (private)")


if __name__ == "__main__":
    main()
