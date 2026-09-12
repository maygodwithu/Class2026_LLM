from contextlib import nullcontext

import numpy as np
import torch

from configs import TrainConfig
from model import GPT, GPTConfig
from train import configure_optimizer, get_batch, get_lr, train_step


# ---- get_batch ----------------------------------------------------------

def test_get_batch_shapes():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, block_size=8, batch_size=4, device="cpu")
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == torch.int64


def test_get_batch_y_is_x_shifted_by_one():
    data = np.arange(1000, dtype=np.uint16)  # data[i] == i, so shifts are easy to check
    x, y = get_batch(data, block_size=8, batch_size=4, device="cpu")
    assert torch.equal(y, x + 1), \
        "y should be x shifted one position later (y[i] == data[start+i+1]) -- this is the next-token target"


# ---- get_lr ---------------------------------------------------------------

def test_get_lr_warmup_increases():
    cfg = TrainConfig(learning_rate=1e-3, min_lr=1e-4, warmup_steps=10)
    lrs = [get_lr(s, cfg, max_steps=100) for s in range(10)]
    assert all(lrs[i] < lrs[i + 1] for i in range(len(lrs) - 1)), \
        "lr should increase monotonically during warmup"
    assert lrs[0] > 0


def test_get_lr_peaks_near_learning_rate_after_warmup():
    cfg = TrainConfig(learning_rate=1e-3, min_lr=1e-4, warmup_steps=10)
    lr_at_warmup_end = get_lr(10, cfg, max_steps=100)
    assert abs(lr_at_warmup_end - cfg.learning_rate) < 1e-4


def test_get_lr_decays_after_warmup():
    cfg = TrainConfig(learning_rate=1e-3, min_lr=1e-4, warmup_steps=10)
    lr_mid = get_lr(50, cfg, max_steps=100)
    lr_end = get_lr(100, cfg, max_steps=100)
    assert lr_end < lr_mid < cfg.learning_rate
    assert lr_end >= cfg.min_lr - 1e-8


def test_get_lr_past_max_steps_returns_min_lr():
    cfg = TrainConfig(learning_rate=1e-3, min_lr=1e-4, warmup_steps=10)
    assert abs(get_lr(200, cfg, max_steps=100) - cfg.min_lr) < 1e-8


# ---- configure_optimizer ---------------------------------------------------

def test_configure_optimizer_splits_by_dim():
    cfg = TrainConfig(weight_decay=0.1, learning_rate=1e-3)
    gpt_cfg = GPTConfig(vocab_size=20, n_embd=8, n_layer=1, n_head=2, block_size=8, dropout=0.0)
    model = GPT(gpt_cfg)
    opt = configure_optimizer(model, cfg)

    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 2
    wds = sorted(g["weight_decay"] for g in opt.param_groups)
    assert wds == [0.0, cfg.weight_decay], \
        "expected one param group with weight_decay=0.0 and one with cfg.weight_decay"

    decay_group = next(g for g in opt.param_groups if g["weight_decay"] == cfg.weight_decay)
    no_decay_group = next(g for g in opt.param_groups if g["weight_decay"] == 0.0)
    assert all(p.dim() >= 2 for p in decay_group["params"]), "decay group should only contain 2D+ params"
    assert all(p.dim() < 2 for p in no_decay_group["params"]), "no-decay group should only contain <2D params"


# ---- train_step (integration) ---------------------------------------------

def test_train_step_reduces_loss_over_many_steps():
    """A tiny model on tiny synthetic data should be able to reduce its
    training loss noticeably within a couple hundred steps -- this is the
    same thing you'll watch happen for real in the server smoke-test run,
    just compressed to run on CPU in seconds for grading."""
    torch.manual_seed(0)
    gpt_cfg = GPTConfig(vocab_size=20, n_embd=16, n_layer=2, n_head=2, block_size=8, dropout=0.0)
    model = GPT(gpt_cfg)
    train_cfg = TrainConfig(
        batch_size=16, learning_rate=1e-2, min_lr=1e-3, warmup_steps=5,
        weight_decay=0.0, grad_clip=1.0, grad_accum_steps=1,
    )
    # A repeating pattern, not i.i.d. random noise: there has to be
    # something predictable in the data for training loss to be able to
    # go down at all. (Purely random tokens have an irreducible loss of
    # ln(vocab_size) -- no amount of training beats that.)
    rng = np.random.RandomState(0)
    pattern = rng.randint(0, gpt_cfg.vocab_size, size=6)
    data = np.tile(pattern, 5000 // len(pattern) + 1)[:5000].astype(np.uint16)
    optimizer = configure_optimizer(model, train_cfg)
    ctx = nullcontext()

    first_loss = train_step(model, optimizer, data, gpt_cfg, train_cfg, "cpu", ctx)
    for _ in range(200):
        last_loss = train_step(model, optimizer, data, gpt_cfg, train_cfg, "cpu", ctx)

    assert last_loss < first_loss - 0.3, \
        f"expected loss to drop meaningfully (first={first_loss:.3f}, last={last_loss:.3f})"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
