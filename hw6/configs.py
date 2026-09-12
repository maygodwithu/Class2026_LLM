"""
Named model presets for the TinyStories proof-of-concept.

Each preset was chosen by hand-computing param count with:
    total = V*E + T*E + L*(12*E^2 + 13*E) + 2*E
(V=vocab_size, E=n_embd, L=n_layer, T=block_size; assumes weight-tied
token embedding / output head, see model.py). The point of keeping four
presets instead of one is pedagogical: at this parameter scale, vocab_size
alone can dominate the whole budget (see `vocab_embed_share` below), which
is a concrete, measurable illustration of a design tradeoff students will
hit immediately once they pick their own tokenizer.
"""

from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    name: str
    tokenizer: str  # "char" or "bpe"
    vocab_size: int  # for "bpe" this is a target passed to the trainer; actual may differ slightly
    n_embd: int
    n_layer: int
    n_head: int
    block_size: int
    dropout: float = 0.1
    bias: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    smoke_test_max_steps: int = 500
    real_run_max_steps: int = 20000
    eval_interval: int = 100
    eval_iters: int = 50
    log_interval: int = 20


MODEL_CONFIGS = {
    "tiny_1m": ModelConfig(
        name="tiny_1m", tokenizer="bpe", vocab_size=4000,
        n_embd=96, n_layer=6, n_head=6, block_size=256,
    ),
    "small_6m": ModelConfig(
        name="small_6m", tokenizer="bpe", vocab_size=6000,
        n_embd=256, n_layer=6, n_head=8, block_size=256,
    ),
    "medium_14m": ModelConfig(
        name="medium_14m", tokenizer="bpe", vocab_size=8000,
        n_embd=384, n_layer=6, n_head=8, block_size=256,
    ),
    "char_cmp_6m": ModelConfig(
        name="char_cmp_6m", tokenizer="char", vocab_size=96,
        n_embd=288, n_layer=6, n_head=8, block_size=512,
    ),
}

TRAIN_CONFIG = TrainConfig()


def estimate_params(cfg: ModelConfig) -> int:
    V, E, L, T = cfg.vocab_size, cfg.n_embd, cfg.n_layer, cfg.block_size
    return V * E + T * E + L * (12 * E * E + 13 * E) + 2 * E


if __name__ == "__main__":
    for name, cfg in MODEL_CONFIGS.items():
        n = estimate_params(cfg)
        vocab_share = (cfg.vocab_size * cfg.n_embd) / n
        print(f"{name:>12}: ~{n:,} params  (vocab-embed share: {vocab_share:.1%})  {asdict(cfg)}")
