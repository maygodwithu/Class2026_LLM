"""
HW5 -- The Training Loop

Goal: implement the four pieces that turn a randomly-initialized model
into one that actually predicts TinyStories text: batching, the learning
rate schedule, the optimizer, and a single training step.

model.py (the GPT you built across HW1-4), configs.py and data/ are all
given -- complete, working versions -- so this assignment is entirely
about the *training mechanics* around the model, not the model itself.

TODO (4 functions): get_batch, get_lr, configure_optimizer, train_step
Everything else (evaluation, logging, checkpointing, the CLI) is given.
"""

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from configs import MODEL_CONFIGS, TRAIN_CONFIG, estimate_params
from data import load_meta
from model import GPT, GPTConfig


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str):
    """
    Sample `batch_size` random windows of length `block_size` from `data`
    (a 1D array of token ids), and build the (input, target) pair for
    next-token prediction.

    Returns: (x, y), each of shape (batch_size, block_size), moved to `device`.

    Steps:
      1. Pick `batch_size` random starting indices `i` such that both
         data[i : i+block_size] and data[i+1 : i+1+block_size] are valid
         (i.e. i ranges over [0, len(data) - block_size - 1)).
         Use np.random.randint(0, len(data) - block_size - 1, size=batch_size).
      2. x's row for a given i is data[i : i+block_size]
      3. y's row for the SAME i is data[i+1 : i+1+block_size] -- y is
         literally x shifted one position to the right. This is what
         "next-token prediction" means: at every position t, the target
         is the token that comes right after it.
      4. Convert each row to a torch.LongTensor (data is uint16, cast with
         .astype(np.int64) first) and torch.stack the rows into (batch_size, block_size).
      5. Move both to `device`.

    Hint: torch.from_numpy(data[i:i+block_size].astype(np.int64))
    """
    # TODO: implement
    raise NotImplementedError


def get_lr(step: int, cfg, max_steps: int) -> float:
    """
    Linear warmup for cfg.warmup_steps, then cosine decay from
    cfg.learning_rate down to cfg.min_lr over the remaining steps.

    Steps:
      1. If step < cfg.warmup_steps: linearly ramp up from 0 to
         cfg.learning_rate. A common formula:
             cfg.learning_rate * (step + 1) / cfg.warmup_steps
      2. If step > max_steps: just return cfg.min_lr (we're past the
         schedule entirely).
      3. Otherwise (in the decay region): compute how far through the
         decay we are as a 0..1 ratio,
             decay_ratio = (step - cfg.warmup_steps) / (max_steps - cfg.warmup_steps)
         then use a cosine curve to interpolate between learning_rate and
         min_lr:
             coeff = 0.5 * (1 + cos(pi * decay_ratio))   # 1 -> 0 as decay_ratio goes 0 -> 1
             return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    Why warmup: starting at full LR on a randomly-initialized model tends
    to cause an early, large, destabilizing update. Why cosine decay
    (rather than, say, a constant LR): it lets the model take large steps
    early (when it's far from a good solution) and small, careful steps
    late (when it's close), without needing a hand-tuned schedule.
    """
    # TODO: implement
    raise NotImplementedError


def configure_optimizer(model: torch.nn.Module, cfg):
    """
    Build an AdamW optimizer with weight decay applied ONLY to 2D+
    parameters (embedding tables, Linear weight matrices) -- not to 1D
    parameters (LayerNorm weight, biases).

    Steps:
      1. Split model.parameters() into two lists based on p.dim() >= 2
         vs p.dim() < 2.
      2. Build param groups:
             [{"params": decay_params, "weight_decay": cfg.weight_decay},
              {"params": no_decay_params, "weight_decay": 0.0}]
      3. Return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.95))

    Why split like this: weight decay pulls parameter values toward zero
    as a regularizer. That makes sense for the big weight matrices doing
    the model's "capacity" work, but decaying LayerNorm's scale or a
    bias term doesn't serve the same purpose and empirically tends to
    hurt more than it helps.
    """
    # TODO: implement
    raise NotImplementedError


def train_step(model, optimizer, train_data, model_cfg, train_cfg, device, ctx) -> float:
    """
    Run exactly one optimizer step (with gradient accumulation) and
    return the scalar loss value (float) for logging.

    Steps:
      1. optimizer.zero_grad(set_to_none=True)
      2. Repeat train_cfg.grad_accum_steps times:
           a. x, y = get_batch(train_data, model_cfg.block_size, train_cfg.batch_size, device)
           b. with ctx:  (the autocast context passed in)
                  logits, loss = model(x, y)
                  loss = loss / train_cfg.grad_accum_steps   # so grads average correctly
           c. loss.backward()
      3. torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
      4. optimizer.step()
      5. return the LAST micro-batch's loss.item() * train_cfg.grad_accum_steps
         (undoing the division from 2b, so the returned number is a normal
         per-example loss value for logging/plotting)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Everything below this line is given -- evaluation, logging, checkpointing,
# and the CLI that wires your four functions above together into a full
# training run.
# ---------------------------------------------------------------------------

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(MODEL_CONFIGS.keys()), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.config]
    train_cfg = TRAIN_CONFIG
    max_steps = args.max_steps or train_cfg.smoke_test_max_steps
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    out_dir = Path(args.out_dir or f"out/{args.config}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(args.data_dir)
    gpt_cfg = GPTConfig(
        vocab_size=meta["vocab_size"], n_embd=model_cfg.n_embd, n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head, block_size=model_cfg.block_size,
        dropout=model_cfg.dropout, bias=model_cfg.bias,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config: {args.config}  device: {device}  max_steps: {max_steps}")
    print(f"estimated params: {estimate_params(model_cfg):,}")

    train_data = np.memmap(Path(args.data_dir) / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(Path(args.data_dir) / "val.bin", dtype=np.uint16, mode="r")
    print(f"train tokens: {len(train_data):,}  val tokens: {len(val_data):,}")

    model = GPT(gpt_cfg).to(device)
    print(f"actual params: {model.get_num_params(non_embedding_position_only=False):,}")

    optimizer = configure_optimizer(model, train_cfg)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    best_val_loss = float("inf")
    t_start = time.time()

    for step in range(max_steps + 1):
        lr = get_lr(step, train_cfg, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % train_cfg.eval_interval == 0 or step == max_steps:
            losses = estimate_loss(
                model, {"train": train_data, "val": val_data},
                model_cfg.block_size, train_cfg.batch_size, device, train_cfg.eval_iters, ctx,
            )
            elapsed = time.time() - t_start
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | "
                  f"lr {lr:.2e} | {elapsed:.1f}s elapsed")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "model_state": model.state_dict(), "gpt_config": asdict(gpt_cfg),
                    "config_name": args.config, "data_dir": str(args.data_dir),
                    "step": step, "val_loss": best_val_loss,
                }, out_dir / "ckpt.pt")

        loss_value = train_step(model, optimizer, train_data, model_cfg, train_cfg, device, ctx)

        if step % train_cfg.log_interval == 0:
            print(f"  step {step:5d} | loss {loss_value:.4f}")

    print(f"done. best val loss: {best_val_loss:.4f}  total time: {time.time() - t_start:.1f}s")
    print(f"checkpoint: {out_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
