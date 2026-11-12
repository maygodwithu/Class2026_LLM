"""
HW5 테스트: python -m pytest test_hw5.py -v -s   (반드시 hw5 폴더 안에서 실행)
data.py(완성본)를 확인하는 테스트는 처음부터 통과하고, 나머지는 TODO를 채워야 통과합니다.
"""
import functools
import math

import pytest
import torch
import torch.nn.functional as F

import data
import tokenizer
import train
from model import GPT, GPTConfig

TEXT = data.load_text()
DEVICE = torch.device("cpu")
V = tokenizer.vocab_size  # 50257


# ---------------------------------------------------------------- 도우미
def tiny_model(seed=0, dropout=0.0):
    torch.manual_seed(seed)
    config = GPTConfig(block_size=16, vocab_size=V, n_embd=32, n_head=2, n_layer=2, dropout=dropout)
    return GPT(config)


def tiny_loaders(chars=None, batch_size=2, block=16):
    text = TEXT if chars is None else TEXT[:chars]
    train_text, val_text = data.split_text(text)
    train_loader = data.create_dataloader(train_text, batch_size, block, block, shuffle=False, drop_last=True)
    val_loader = data.create_dataloader(val_text, batch_size, block, block, shuffle=False, drop_last=False)
    return train_loader, val_loader


@functools.lru_cache(maxsize=None)
def book_model():
    """교재 5장 모델: 124M, 문맥 길이 256, dropout 0.1, seed 123 (한 번만 만든다)"""
    torch.manual_seed(123)
    model = GPT(GPTConfig(block_size=256, vocab_size=V, dropout=0.1))
    model.eval()
    return model


# ---------------------------------------------------------------- data.py (완성본: 처음부터 통과)
def test_dataset_target_is_input_shifted_by_one():
    ds = data.GPTDataset(TEXT, max_length=8, stride=8)
    x, y = ds[0]
    ids = tokenizer.encode(TEXT)
    assert x.shape == (8,) and y.shape == (8,)
    assert x.tolist() == ids[:8]
    assert y.tolist() == ids[1:9]
    assert torch.equal(x[1:], y[:-1])  # 정답은 입력을 한 칸 민 것


@pytest.mark.parametrize("max_length,stride", [(4, 4), (4, 2), (8, 3)])
def test_dataset_stride(max_length, stride):
    ids = tokenizer.encode(TEXT)
    ds = data.GPTDataset(TEXT, max_length, stride)
    assert len(ds) == len(range(0, len(ids) - max_length, stride))
    assert ds[1][0].tolist() == ids[stride : stride + max_length]


def test_split_text():
    train_text, val_text = data.split_text(TEXT, 0.9)
    assert train_text + val_text == TEXT
    assert len(train_text) == int(0.9 * len(TEXT))


def test_book_loader_sizes():
    train_text, val_text = data.split_text(TEXT)
    train_loader = data.create_dataloader(train_text, 2, 256, 256, shuffle=False, drop_last=True)
    val_loader = data.create_dataloader(val_text, 2, 256, 256, shuffle=False, drop_last=False)
    x, y = next(iter(train_loader))
    assert x.shape == (2, 256) and y.shape == (2, 256)
    assert (len(train_loader), len(val_loader)) == (9, 1)
    print(f"\n[data] 전체 {len(tokenizer.encode(TEXT))}토큰, 학습 배치 {len(train_loader)}개, 검증 배치 {len(val_loader)}개")


# ---------------------------------------------------------------- TODO 1: calc_loss_batch
def test_calc_loss_batch_matches_definition():
    model = tiny_model()
    x, y = next(iter(tiny_loaders()[0]))
    loss = train.calc_loss_batch(x, y, model, DEVICE)
    logp = torch.log_softmax(model(x), dim=-1)  # (b, t, V)
    manual = -logp.gather(-1, y.unsqueeze(-1)).mean()  # 위치마다 -log(정답 확률)의 평균
    assert torch.allclose(loss, manual, atol=1e-5)


def test_calc_loss_batch_is_scalar_and_has_gradient():
    model = tiny_model()
    x, y = next(iter(tiny_loaders()[0]))
    loss = train.calc_loss_batch(x, y, model, DEVICE)
    assert loss.dim() == 0
    assert loss.requires_grad
    loss.backward()
    assert model.out_head.weight.grad is not None


def test_loss_is_small_for_correct_logits_and_large_for_wrong_ones():
    x, y = next(iter(tiny_loaders()[0]))
    right = lambda inp: F.one_hot(y, V).float() * 20.0  # 정답 위치에 높은 점수
    wrong = lambda inp: F.one_hot(torch.roll(y, 1, dims=1), V).float() * 20.0  # 엉뚱한 위치
    assert train.calc_loss_batch(x, y, right, DEVICE).item() < 1e-3
    assert train.calc_loss_batch(x, y, wrong, DEVICE).item() > 10


def test_initial_loss_is_near_log_vocab():
    model = tiny_model()
    model.eval()
    x, y = next(iter(tiny_loaders()[0]))
    loss = train.calc_loss_batch(x, y, model, DEVICE).item()
    print(f"\n[loss] 학습 전 손실 {loss:.3f}  (ln({V}) = {math.log(V):.3f}: 5만 개 중 무작위로 고르는 수준)")
    assert math.log(V) - 0.5 < loss < math.log(V) + 1.0


def test_calc_loss_loader_is_average_over_batches():
    model = tiny_model()
    model.eval()
    train_loader, _ = tiny_loaders(chars=4000)
    losses = [train.calc_loss_batch(x, y, model, DEVICE).item() for x, y in train_loader]
    assert train.calc_loss_loader(train_loader, model, DEVICE) == pytest.approx(sum(losses) / len(losses), abs=1e-5)
    assert train.calc_loss_loader(train_loader, model, DEVICE, num_batches=2) == pytest.approx(sum(losses[:2]) / 2, abs=1e-5)


# ---------------------------------------------------------------- TODO 2: perplexity
def test_perplexity_values():
    assert float(train.perplexity(torch.tensor(0.0))) == pytest.approx(1.0)
    assert float(train.perplexity(torch.tensor(math.log(10)))) == pytest.approx(10.0, rel=1e-5)
    assert float(train.perplexity(torch.tensor(math.log(V)))) == pytest.approx(V, rel=1e-4)


def test_perplexity_increases_with_loss():
    p = [float(train.perplexity(torch.tensor(v))) for v in (1.0, 2.0, 5.0)]
    assert p[0] < p[1] < p[2]


# ---------------------------------------------------------------- TODO 3: evaluate_model
def test_evaluate_model_turns_dropout_off():
    model = tiny_model(dropout=0.5)
    model.train()
    train_loader, val_loader = tiny_loaders(chars=4000)
    first = train.evaluate_model(model, train_loader, val_loader, DEVICE, eval_iter=3)
    second = train.evaluate_model(model, train_loader, val_loader, DEVICE, eval_iter=3)
    assert first == second  # dropout이 꺼져 있으면 같은 결과, 켜져 있으면 매번 다름


def test_evaluate_model_matches_eval_mode_loss_and_restores_train_mode():
    model = tiny_model(dropout=0.5)
    model.train()
    train_loader, val_loader = tiny_loaders(chars=4000)
    train_loss, val_loss = train.evaluate_model(model, train_loader, val_loader, DEVICE, eval_iter=3)
    assert model.training  # 평가 뒤에는 학습 모드로 되돌아와야 한다
    model.eval()
    with torch.no_grad():
        ref_train = train.calc_loss_loader(train_loader, model, DEVICE, num_batches=3)
        ref_val = train.calc_loss_loader(val_loader, model, DEVICE, num_batches=3)
    assert train_loss == pytest.approx(ref_train, abs=1e-6)
    assert val_loss == pytest.approx(ref_val, abs=1e-6)
    assert isinstance(train_loss, float) and isinstance(val_loss, float)


# ---------------------------------------------------------------- TODO 4: 학습 루프 (backward, step)
def run_training(model, num_epochs, eval_freq=10**6, lr=1e-3, chars=4000):
    train_loader, val_loader = tiny_loaders(chars=chars)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    logs = train.train_model_simple(
        model, train_loader, val_loader, optimizer, DEVICE, num_epochs=num_epochs,
        eval_freq=eval_freq, eval_iter=2, start_context="Every effort")
    return logs, train_loader


def test_training_updates_weights_and_has_gradients():
    model = tiny_model(1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    run_training(model, num_epochs=1)
    assert model.out_head.weight.grad is not None  # backward()가 불렸다
    assert not torch.equal(before["out_head.weight"], model.out_head.weight)  # step()이 불렸다
    assert not torch.equal(before["trf_blocks.0.ff.layers.0.weight"], model.trf_blocks[0].ff.layers[0].weight)


def test_train_model_simple_matches_manual_loop():
    epochs = 2
    m1 = tiny_model(1)
    run_training(m1, num_epochs=epochs)

    m2 = tiny_model(1)
    train_loader, _ = tiny_loaders(chars=4000)
    opt = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    for _ in range(epochs):
        m2.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(m2(x).flatten(0, 1), y.flatten())
            loss.backward()
            opt.step()
    for (name, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), name


def test_train_model_simple_returns_logs():
    model = tiny_model(1)
    epochs, eval_freq = 2, 3
    (train_losses, val_losses, tokens_seen), train_loader = run_training(model, epochs, eval_freq=eval_freq)
    steps = epochs * len(train_loader)
    n_eval = len(range(0, steps, eval_freq))
    assert len(train_losses) == len(val_losses) == len(tokens_seen) == n_eval
    assert tokens_seen[0] == 2 * 16  # 첫 스텝 뒤: 배치 2 x 길이 16
    assert all(a < b for a, b in zip(tokens_seen, tokens_seen[1:]))


def test_training_reduces_loss():
    model = tiny_model(1)
    (train_losses, val_losses, _), _ = run_training(model, num_epochs=3, eval_freq=5, chars=None)
    print(f"\n[train] 작은 모델 3에폭: train 손실 {train_losses[0]:.3f} -> {train_losses[-1]:.3f}")
    assert train_losses[-1] < train_losses[0] - 1.0


def test_model_can_memorize_one_batch():
    model = tiny_model(1)
    x, y = next(iter(tiny_loaders()[0]))
    batch = [(x, y)]  # 같은 배치 하나를 계속 보여준다
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = train.calc_loss_loader(batch, model, DEVICE)
    train.train_model_simple(model, batch, batch, optimizer, DEVICE, num_epochs=80,
                             eval_freq=10**6, eval_iter=1, start_context="Every effort")
    last = train.calc_loss_loader(batch, model, DEVICE)
    print(f"\n[train] 배치 하나 외우기: 손실 {first:.3f} -> {last:.3f}")
    assert last < 1.0


def test_generate_and_print_sample(capsys):
    model = tiny_model()
    model.train()
    train.generate_and_print_sample(model, DEVICE, "Every effort moves you", max_new_tokens=5)
    out = capsys.readouterr().out
    assert out.startswith("Every effort moves you")
    assert model.training


# ---------------------------------------------------------------- 교재 5.1의 값 (124M 모델)
def test_book_initial_loss():
    model = book_model()
    train_text, val_text = data.split_text(TEXT)
    train_loader = data.create_dataloader(train_text, 2, 256, 256, shuffle=False, drop_last=True)
    val_loader = data.create_dataloader(val_text, 2, 256, 256, shuffle=False, drop_last=False)
    with torch.no_grad():
        train_loss = train.calc_loss_loader(train_loader, model, DEVICE)
        val_loss = train.calc_loss_loader(val_loader, model, DEVICE)
    print(f"\n[book] 학습 전 124M 모델: train 손실 {train_loss:.4f}, val 손실 {val_loss:.4f}")
    print(f"       perplexity: {float(train.perplexity(torch.tensor(train_loss))):.0f} (어휘 수 {V}와 비슷)")
    assert train_loss == pytest.approx(10.9876, abs=0.02)  # 교재: 10.9876
    assert val_loss == pytest.approx(10.9811, abs=0.02)  # 교재: 10.9811
