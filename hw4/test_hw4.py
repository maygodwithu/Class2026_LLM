import functools

import torch
import torch.nn as nn

import tokenizer
from model import GPT, GPTConfig, FeedForward, TransformerBlock

E, H, T = 16, 2, 6


def cfg(dropout=0.0, n_embd=E, n_head=H, block_size=T, n_layer=2):
    return GPTConfig(vocab_size=50, n_embd=n_embd, n_head=n_head, block_size=block_size,
                     n_layer=n_layer, dropout=dropout)


@functools.lru_cache(maxsize=None)
def book_model():
    # 교재 4.6절의 GPT_CONFIG_124M (124M 모델), seed 123, 드롭아웃 0.1
    torch.manual_seed(123)
    config = GPTConfig(vocab_size=50257, block_size=1024, n_embd=768, n_head=12, n_layer=12, dropout=0.1)
    return GPT(config).eval()


def make_block():
    torch.manual_seed(0)
    return TransformerBlock(cfg()), torch.randn(2, T, E)


def zero_(*linears):
    with torch.no_grad():
        for lin in linears:
            lin.weight.zero_()
            if lin.bias is not None:
                lin.bias.zero_()


# ---------------- 1단계: FeedForward (교재 4.3) ----------------

def test_feedforward_structure():
    ff = FeedForward(cfg())
    lin1, act, lin2 = ff.layers[0], ff.layers[1], ff.layers[2]
    assert isinstance(lin1, nn.Linear) and isinstance(act, nn.GELU) and isinstance(lin2, nn.Linear)
    assert (lin1.in_features, lin1.out_features) == (E, 4 * E)
    assert (lin2.in_features, lin2.out_features) == (4 * E, E)


def test_feedforward_param_count():
    ff = FeedForward(cfg())
    assert sum(p.numel() for p in ff.parameters()) == 8 * E * E + 5 * E


def test_book_example_feedforward():
    ffn = FeedForward(cfg(n_embd=768, n_head=12, block_size=1024))
    out = ffn(torch.rand(2, 3, 768))  # 교재 4.3절: [batch_size, num_token, emb_size]
    print("\n출력 크기:", tuple(out.shape))
    assert out.shape == (2, 3, 768)


def test_feedforward_positions_are_independent():
    # attention은 토큰들을 섞지만, FFN은 각 위치를 따로따로 처리함
    torch.manual_seed(0)
    ff = FeedForward(cfg())
    x = torch.randn(2, T, E)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(2, T - 3, E)
    assert torch.allclose(ff(x)[:, :3], ff(x2)[:, :3], atol=1e-6)


# ---------------- 2단계: TransformerBlock -- 어텐션 블록 + shortcut (교재 4.4, 4.5) ----------------

def test_attention_half_formula():
    block, x = make_block()
    block(x)
    expected = x + block.att(block.norm1(x))
    assert torch.allclose(block.after_attention, expected, atol=1e-6)


def test_attention_half_is_shortcut():
    # attention의 마지막 층을 0으로 만들면 어텐션 블록은 입력을 그대로 통과시켜야 함
    block, x = make_block()
    zero_(block.att.out_proj)
    block(x)
    assert torch.allclose(block.after_attention, x, atol=1e-6)


def test_ff_half_formula():
    block, x = make_block()
    out = block(x)
    y = block.after_attention
    assert torch.allclose(out, y + block.ff(block.norm2(y)), atol=1e-6)


def test_block_output_shape():
    block, x = make_block()
    assert block(x).shape == x.shape


def test_book_example_block():
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)  # 교재 4.5절: [batch_size, num_tokens, emb_dim]
    block = TransformerBlock(cfg(n_embd=768, n_head=12, block_size=1024))
    out = block(x)
    print("\n입력 크기:", tuple(x.shape), "\n출력 크기:", tuple(out.shape))
    assert out.shape == (2, 4, 768)


def test_block_is_identity_when_sublayers_are_zero():
    block, x = make_block()
    zero_(block.att.out_proj, block.ff.layers[2])
    assert torch.allclose(block(x), x, atol=1e-6)


def test_shortcut_passes_gradient():
    # sublayer 출력이 0이어도 shortcut이 있으면 입력까지 gradient가 그대로 흐름 (교재 4.4절)
    block, x = make_block()
    zero_(block.att.out_proj, block.ff.layers[2])
    x.requires_grad_(True)
    block(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_future_token_does_not_change_past():
    block, x = make_block()
    out1 = block(x)
    x2 = x.clone()
    x2[:, -1] = torch.randn(2, E)
    out2 = block(x2)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)


# ---------------- 3단계: GPT 구조 -- 블록 쌓기 (교재 4.6) ----------------

def small_gpt(n_layer=3, block_size=8, vocab_size=100):
    return GPT(GPTConfig(vocab_size=vocab_size, n_embd=E, n_head=H, block_size=block_size,
                         n_layer=n_layer, dropout=0.0)).eval()


def test_gpt_has_stacked_blocks():
    m = small_gpt(n_layer=3)
    blocks = list(m.trf_blocks)
    assert len(blocks) == 3
    assert all(isinstance(b, TransformerBlock) for b in blocks)
    assert len({id(b) for b in blocks}) == 3, "같은 블록 하나를 반복해서 넣으면 안 됩니다 (층마다 따로 학습되어야 함)"


def test_gpt_final_norm_and_head():
    m = small_gpt()
    assert isinstance(m.final_norm, nn.LayerNorm)
    assert isinstance(m.out_head, nn.Linear)
    assert (m.out_head.in_features, m.out_head.out_features) == (E, 100)
    assert m.out_head.bias is None


def test_book_param_count():
    model = book_model()
    total = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터 개수: {total:,}")
    assert total == 163_009_536  # 교재 4.6절
    without_head = total - sum(p.numel() for p in model.out_head.parameters())
    print(f"가중치 묶기를 고려한 파라미터 개수: {without_head:,}")
    assert without_head == 124_412_160


# ---------------- 4단계: GPT.forward -- logits (교재 4.6) ----------------

def test_gpt_logits_shape():
    m = small_gpt()
    out = m(torch.randint(0, 100, (2, 5)))
    assert out.shape == (2, 5, 100)


def test_gpt_forward_composition():
    # logits = out_head(final_norm(trf_blocks(임베딩)))
    m = small_gpt()
    idx = torch.randint(0, 100, (2, 6))
    x = m.transformer.wte(idx) + m.transformer.wpe(torch.arange(6))
    expected = m.out_head(m.final_norm(m.trf_blocks(x)))
    assert torch.allclose(m(idx), expected, atol=1e-5)


def test_all_parameters_get_gradients():
    # 블록, final_norm, out_head, 임베딩 중 연결이 빠진 것이 없는지 확인
    m = small_gpt()
    m.train()
    m(torch.randint(0, 100, (2, 6))).sum().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"gradient가 흐르지 않는 파라미터: {missing[:3]}"


def test_gpt_is_causal():
    torch.manual_seed(0)
    m = small_gpt()
    idx = torch.randint(0, 100, (2, 8))
    out1 = m(idx)
    idx2 = idx.clone()
    idx2[:, -1] = (idx2[:, -1] + 1) % 100  # 마지막 토큰만 바꿈
    out2 = m(idx2)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
    assert not torch.allclose(out1[:, -1], out2[:, -1], atol=1e-5)


def test_book_example_logits():
    batch = torch.tensor([[6109, 3626, 6100, 345],    # "Every effort moves you"
                          [6109, 1110, 6622, 257]])   # "Every day holds a"
    out = book_model()(batch)
    print("\n입력 배치:\n", batch, "\n출력 크기:", tuple(out.shape))
    assert out.shape == (2, 4, 50257)


def test_text_to_gpt_end_to_end():
    ids = tokenizer.encode("Once upon a time, there was a small robot.")
    idx = torch.tensor([ids])
    c = GPTConfig(vocab_size=tokenizer.vocab_size, n_embd=32, n_head=4, block_size=64, n_layer=2)
    y = GPT(c).eval()(idx)
    assert y.shape == (1, len(ids), tokenizer.vocab_size)


# ---------------- 5단계: generate -- 그리디 디코딩 (교재 4.7) ----------------

def test_generate_length_and_prefix():
    m = small_gpt()
    idx = torch.randint(0, 100, (2, 3))
    out = m.generate(idx, max_new_tokens=4)
    assert out.shape == (2, 7)
    assert torch.equal(out[:, :3], idx)


def test_generate_is_greedy():
    m = small_gpt()
    idx = torch.randint(0, 100, (2, 4))
    out = m.generate(idx, max_new_tokens=1)
    assert torch.equal(out[:, -1], m(idx)[:, -1].argmax(dim=-1))


def test_generate_is_deterministic():
    m = small_gpt()
    idx = torch.randint(0, 100, (1, 3))
    assert torch.equal(m.generate(idx, 5), m.generate(idx, 5))


def test_generate_crops_context():
    # block_size(8)보다 길게 생성해도 동작하고, 매 단계 마지막 8개 토큰만 보고 고른 것과 같아야 함
    torch.manual_seed(0)
    m = small_gpt(block_size=8)
    idx = torch.randint(0, 100, (1, 8))
    out = m.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 13)
    ref = idx
    for _ in range(5):
        nxt = m(ref[:, -8:])[:, -1].argmax(dim=-1, keepdim=True)
        ref = torch.cat((ref, nxt), dim=1)
    assert torch.equal(out, ref)


def test_book_example_generate():
    start = "Hello, I am"
    ids = tokenizer.encode(start)
    out = book_model().generate(torch.tensor(ids).unsqueeze(0), max_new_tokens=6)
    print("\n인코딩된 ID:", ids)
    print("출력:", out)
    print("출력 길이:", len(out[0]))
    print("출력 텍스트:", tokenizer.decode(out.squeeze(0).tolist()))
    # 학습 전 모델이라 엉터리 문장이 나오는 것이 정상. 교재 코드(gpt.py)와 층을 만드는 순서가 같으면
    # "Hello, I am Featureiman Byeswickattribute argue"가 나옵니다 (순서가 다르면 초기값이 달라 다른 문장).
    assert ids == [15496, 11, 314, 716]
    assert out.shape == (1, 10)
    assert out[0, :4].tolist() == ids


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
