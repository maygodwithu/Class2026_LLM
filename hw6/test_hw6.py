from contextlib import nullcontext

import numpy as np
import torch

from configs import TrainConfig
from model import GPT, GPTConfig
from finetune import configure_finetune_optimizer, freeze_backbone, train_step


def make_cfg():
    return GPTConfig(vocab_size=20, n_embd=16, n_layer=4, n_head=2, block_size=8, dropout=0.0)


# ---- freeze_backbone -------------------------------------------------------

def test_freeze_backbone_freezes_only_early_blocks():
    cfg = make_cfg()
    m = GPT(cfg)
    freeze_backbone(m, n_finetune_layers=1)

    for block in m.blocks[:-1]:
        assert all(not p.requires_grad for p in block.parameters()), \
            "early blocks should be frozen"
    assert all(p.requires_grad for p in m.blocks[-1].parameters()), \
        "the last n_finetune_layers blocks should stay trainable"


def test_freeze_backbone_leaves_embeddings_and_head_trainable():
    cfg = make_cfg()
    m = GPT(cfg)
    freeze_backbone(m, n_finetune_layers=1)

    assert m.wte.weight.requires_grad, \
        "wte should NOT be frozen (it's tied to lm_head -- freezing it would silently freeze lm_head too)"
    assert m.wpe.weight.requires_grad
    assert all(p.requires_grad for p in m.ln_f.parameters())
    assert m.lm_head.weight.requires_grad


def test_freeze_backbone_n_finetune_layers_2():
    cfg = make_cfg()  # n_layer=4
    m = GPT(cfg)
    freeze_backbone(m, n_finetune_layers=2)
    for block in m.blocks[:2]:
        assert all(not p.requires_grad for p in block.parameters())
    for block in m.blocks[2:]:
        assert all(p.requires_grad for p in block.parameters())


# ---- configure_finetune_optimizer -----------------------------------------

def test_optimizer_excludes_frozen_params():
    cfg = make_cfg()
    m = GPT(cfg)
    freeze_backbone(m, n_finetune_layers=1)
    train_cfg = TrainConfig(weight_decay=0.1, learning_rate=1e-4)
    opt = configure_finetune_optimizer(m, train_cfg)

    opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    frozen_ids = {id(p) for p in m.blocks[0].parameters()}
    trainable_ids = {id(p) for p in m.parameters() if p.requires_grad}

    assert opt_param_ids.isdisjoint(frozen_ids), "optimizer must not receive frozen parameters"
    assert opt_param_ids == trainable_ids, "optimizer should receive exactly the trainable parameters"


def test_optimizer_weight_decay_grouping_still_applies():
    cfg = make_cfg()
    m = GPT(cfg)
    freeze_backbone(m, n_finetune_layers=1)
    train_cfg = TrainConfig(weight_decay=0.1, learning_rate=1e-4)
    opt = configure_finetune_optimizer(m, train_cfg)

    wds = sorted(g["weight_decay"] for g in opt.param_groups)
    assert wds == [0.0, train_cfg.weight_decay]


# ---- integration: frozen params truly don't move ---------------------------

def test_finetune_updates_only_unfrozen_params():
    torch.manual_seed(0)
    cfg = make_cfg()
    m = GPT(cfg)
    freeze_backbone(m, n_finetune_layers=1)

    before_frozen = [p.clone() for p in m.blocks[0].parameters()]
    before_trainable = [p.clone() for p in m.blocks[-1].parameters()]

    train_cfg = TrainConfig(
        batch_size=8, learning_rate=1e-2, min_lr=1e-3, warmup_steps=2,
        weight_decay=0.0, grad_clip=1.0, grad_accum_steps=1,
    )
    rng = np.random.RandomState(0)
    pattern = rng.randint(0, cfg.vocab_size, size=6)
    data = np.tile(pattern, 2000 // 6 + 1)[:2000].astype(np.uint16)

    optimizer = configure_finetune_optimizer(m, train_cfg)
    ctx = nullcontext()
    for _ in range(30):
        train_step(m, optimizer, data, cfg, train_cfg, "cpu", ctx)

    after_frozen = list(m.blocks[0].parameters())
    after_trainable = list(m.blocks[-1].parameters())

    for b, a in zip(before_frozen, after_frozen):
        assert torch.equal(b, a), "frozen block parameters changed during fine-tuning -- freeze_backbone or " \
                                   "configure_finetune_optimizer isn't excluding them correctly"

    assert any(not torch.equal(b, a) for b, a in zip(before_trainable, after_trainable)), \
        "trainable block parameters should have changed during fine-tuning"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
