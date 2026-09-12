"""
HW6 -- Fine-tuning

Goal: take a model that already learned general English/story structure
from pretraining (HW5), and adapt it to a narrower domain by continuing
training on new data -- while being deliberate about WHICH parameters
are allowed to change.

get_batch, get_lr and train_step are given below exactly as solved in
HW5 (already tested there) -- this assignment is entirely about two new
pieces: freezing part of a pretrained model, and building an optimizer
that only touches the parameters you left trainable.

TODO (2 functions): freeze_backbone, configure_finetune_optimizer
Everything else (loading a checkpoint, the training loop, the CLI) is given.
"""

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from configs import TrainConfig
from data import load_meta
from model import GPT, GPTConfig


def freeze_backbone(model: GPT, n_finetune_layers: int) -> None:
    """
    Freeze every parameter EXCEPT the last `n_finetune_layers` transformer
    blocks (model.blocks[-n_finetune_layers:]).

    Specifically:
      - model.wte, model.wpe: leave trainable (don't touch)
      - model.blocks[: len(model.blocks) - n_finetune_layers]: freeze
        (set .requires_grad = False on every parameter in these blocks)
      - model.blocks[len(model.blocks) - n_finetune_layers :]: leave trainable
      - model.ln_f, model.lm_head: leave trainable

    Why freeze anything at all: the early blocks of a pretrained model
    tend to encode fairly general, reusable structure (local grammar,
    common word patterns); the later blocks are where more task/domain-
    specific behavior tends to live. Freezing the early blocks means
    fewer parameters to update (faster, less prone to overfitting a small
    fine-tuning set) while still letting the model adapt where it matters
    most.

    Note: we deliberately do NOT freeze wte/wpe here. wte is TIED to
    lm_head (see model.py) -- freezing wte would silently also freeze
    lm_head, since they're the exact same tensor. Leaving both trainable
    sidesteps that trap entirely.

    This function returns nothing -- it mutates `model` in place by
    setting .requires_grad on its parameters.
    """
    # TODO: implement
    raise NotImplementedError


def configure_finetune_optimizer(model: torch.nn.Module, cfg):
    """
    Same idea as HW5's configure_optimizer (AdamW, weight decay only on
    2D+ params) -- but this time the model may have some parameters with
    requires_grad=False (from freeze_backbone), and those must NOT be
    handed to the optimizer at all.

    Steps:
      1. Only consider `p for p in model.parameters() if p.requires_grad`
         -- skip frozen parameters entirely (don't just give them
         weight_decay=0, actually exclude them from both param groups).
      2. Same 2D+ vs <2D split as before for the weight_decay grouping.
      3. Return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.95))

    Why this matters beyond "it would waste memory": AdamW keeps running
    moment estimates for every parameter you give it. Handing it frozen
    parameters (which never receive gradients) is harmless numerically
    here, but in general is exactly the kind of bug that silently wastes
    optimizer state and can bite you once you're fine-tuning something
    large enough that the accounting matters.
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Everything below this line is given.
# ---------------------------------------------------------------------------

def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str):
    """Given -- solved in HW5."""
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_lr(step: int, cfg, max_steps: int) -> float:
    """Given -- solved in HW5."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step > max_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def train_step(model, optimizer, train_data, model_cfg, train_cfg, device, ctx) -> float:
    """Given -- solved in HW5."""
    optimizer.zero_grad(set_to_none=True)
    loss_value = None
    for _ in range(train_cfg.grad_accum_steps):
        x, y = get_batch(train_data, model_cfg.block_size, train_cfg.batch_size, device)
        with ctx:
            _, loss = model(x, y)
            loss = loss / train_cfg.grad_accum_steps
        loss.backward()
        loss_value = loss.item() * train_cfg.grad_accum_steps
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
    optimizer.step()
    return loss_value


@torch.no_grad()
def estimate_loss(model, data_splits, block_size, batch_size, device, eval_iters, ctx):
    model.eval()
    out = {}
    for split, data in data_splits.items():
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def load_pretrained(ckpt_path: str, device: str) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    gpt_cfg = GPTConfig(**ckpt["gpt_config"])
    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded base checkpoint: {ckpt_path} (pretrain step {ckpt['step']}, val loss {ckpt['val_loss']:.4f})")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", required=True, help="pretrained checkpoint from HW5 (out/*/ckpt.pt)")
    parser.add_argument("--data-dir", required=True, help="fine-tuning data, prepared with --reuse-tokenizer")
    parser.add_argument("--n-finetune-layers", type=int, default=1)
    parser.add_argument("--out-dir", default="out/finetuned")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                         help="fine-tuning LR is typically much smaller than pretraining LR")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = load_pretrained(args.base_ckpt, device)
    meta = load_meta(args.data_dir)
    assert meta["vocab_size"] == model.cfg.vocab_size, (
        f"fine-tuning data vocab_size ({meta['vocab_size']}) doesn't match the base model's "
        f"({model.cfg.vocab_size}) -- did you forget --reuse-tokenizer when preparing this data?"
    )

    freeze_backbone(model, args.n_finetune_layers)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({n_trainable / n_total:.1%})")

    train_cfg = TrainConfig(
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        min_lr=args.learning_rate / 10, warmup_steps=20, weight_decay=0.1,
    )

    train_data = np.memmap(Path(args.data_dir) / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(Path(args.data_dir) / "val.bin", dtype=np.uint16, mode="r")
    print(f"fine-tune train tokens: {len(train_data):,}  val tokens: {len(val_data):,}")

    optimizer = configure_finetune_optimizer(model, train_cfg)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    best_val_loss = float("inf")
    t_start = time.time()

    losses = estimate_loss(model, {"train": train_data, "val": val_data},
                            model.cfg.block_size, train_cfg.batch_size, device, 20, ctx)
    print(f"before fine-tuning | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    for step in range(args.max_steps + 1):
        lr = get_lr(step, train_cfg, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % 50 == 0 or step == args.max_steps:
            losses = estimate_loss(model, {"train": train_data, "val": val_data},
                                    model.cfg.block_size, train_cfg.batch_size, device, 20, ctx)
            elapsed = time.time() - t_start
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | "
                  f"lr {lr:.2e} | {elapsed:.1f}s elapsed")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "model_state": model.state_dict(), "gpt_config": asdict(model.cfg),
                    "config_name": "finetuned", "data_dir": str(args.data_dir),
                    "step": step, "val_loss": best_val_loss,
                }, out_dir / "ckpt.pt")

        train_step(model, optimizer, train_data, model.cfg, train_cfg, device, ctx)

    print(f"done. best fine-tuned val loss: {best_val_loss:.4f}  total time: {time.time() - t_start:.1f}s")
    print(f"checkpoint: {out_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
