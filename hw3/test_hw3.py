import torch
import torch.nn.functional as F

import tokenizer
from model import GPT, GPTConfig, MultiHeadAttention

D_IN, D_OUT, HEADS, T = 16, 8, 2, 6
HEAD_DIM = D_OUT // HEADS


def make(dropout=0.0):
    torch.manual_seed(0)
    m = MultiHeadAttention(D_IN, D_OUT, T, dropout, num_heads=HEADS)
    x = torch.randn(2, T, D_IN)
    return m, x


# 교재 3장의 예제 입력 (Your journey starts with one step), 배치 크기 2
BOOK_INPUTS = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)
BOOK_BATCH = torch.stack((BOOK_INPUTS, BOOK_INPUTS), dim=0)

# 교재 3.6.2절 출력 (seed 123, d_in=3, d_out=2, num_heads=2) -> 크기 (2, 6, 2)
BOOK_MHA_OUT = torch.tensor(
    [[0.3190, 0.4858],
     [0.2943, 0.3897],
     [0.2856, 0.3593],
     [0.2693, 0.3873],
     [0.2639, 0.3928],
     [0.2575, 0.4028]]
)


# ---------------- 1단계: head 나누기 ----------------

def test_split_shapes():
    m, x = make()
    m(x)
    shape = (2, HEADS, T, HEAD_DIM)
    assert m.queries.shape == m.keys.shape == m.values.shape == shape


def test_split_values():
    # head h는 W(x)의 마지막 차원을 head_dim씩 자른 h번째 조각이어야 함
    m, x = make()
    m(x)
    for h in range(HEADS):
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        assert torch.allclose(m.queries[:, h], m.W_query(x)[..., sl])
        assert torch.allclose(m.keys[:, h], m.W_key(x)[..., sl])
        assert torch.allclose(m.values[:, h], m.W_value(x)[..., sl])


def test_attention_table_shape():
    m, x = make()
    m(x)
    assert m.attn_scores.shape == (2, HEADS, T, T)
    assert m.attn_weights.shape == (2, HEADS, T, T)


def test_attention_rows_sum_to_one():
    m, x = make()
    m(x)
    assert torch.allclose(m.attn_weights.sum(dim=-1), torch.ones(2, HEADS, T), atol=1e-5)


def test_causal_upper_triangle_is_zero_for_every_head():
    m, x = make()
    m(x)
    for h in range(HEADS):
        print(f"\nattention table (첫 번째 문장, head {h}):\n", m.attn_weights[0, h].detach().round(decimals=3))
    assert torch.all(torch.triu(m.attn_weights, diagonal=1) == 0), "미래 위치의 확률이 0이 아닙니다"


def test_heads_attend_differently():
    m, x = make()
    m(x)
    assert not torch.allclose(m.attn_weights[:, 0], m.attn_weights[:, 1])


def test_dropout_is_off_in_eval():
    m, x = make(dropout=0.5)
    m.eval()
    m(x)
    assert torch.allclose(m.attn_weights.sum(dim=-1), torch.ones(2, HEADS, T), atol=1e-5)


# ---------------- 2단계: head 합치기 ----------------

def test_output_shape():
    m, x = make()
    assert m(x).shape == (2, T, D_OUT), "head를 합치면 (배치, 토큰수, d_out)"


def test_merge_matches_manual():
    m, x = make()
    out = m(x)
    ctx = (m.attn_weights @ m.values).transpose(1, 2).reshape(2, T, D_OUT)
    assert torch.allclose(out, m.out_proj(ctx), atol=1e-6)


def test_book_example_multihead():
    torch.manual_seed(123)
    m = MultiHeadAttention(3, 2, 6, 0.0, num_heads=2)
    out = m(BOOK_BATCH)
    print("\ncontext_vecs.shape:", tuple(out.shape))
    assert out.shape == (2, 6, 2)
    assert torch.allclose(out[0], BOOK_MHA_OUT, atol=1e-3)


def test_matches_torch_causal_attention():
    # nanoGPT 실제 코드가 쓰는 F.scaled_dot_product_attention(is_causal=True)와 비교
    m, x = make()
    out = m(x)
    ref = F.scaled_dot_product_attention(m.queries, m.keys, m.values, is_causal=True)
    ref = m.out_proj(ref.transpose(1, 2).reshape(2, T, D_OUT))
    assert torch.allclose(out, ref, atol=1e-5)


def test_future_token_does_not_change_past():
    m, x = make()
    out1 = m(x)
    x2 = x.clone()
    x2[:, -1] = torch.randn(2, D_IN)
    out2 = m(x2)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)


# ---------------- GPT에 연결: text -> tokenizer -> GPT(임베딩 + 어텐션) ----------------

def gpt_cfg():
    return GPTConfig(vocab_size=100, n_embd=16, n_head=2, block_size=8, dropout=0.0)


def test_gpt_has_multihead_attention():
    m = GPT(gpt_cfg())
    assert isinstance(m.att, MultiHeadAttention)
    assert m.att.num_heads == 2


def test_gpt_output_differs_from_embedding():
    torch.manual_seed(0)
    m = GPT(gpt_cfg()).eval()
    idx = torch.randint(0, 100, (2, 8))
    emb = m.transformer.drop(m.transformer.wte(idx) + m.transformer.wpe(torch.arange(8)))
    out = m(idx)
    assert out.shape == emb.shape
    assert not torch.allclose(out, emb), "어텐션을 거친 출력이 임베딩 출력과 같습니다"


def test_gpt_is_causal():
    torch.manual_seed(0)
    m = GPT(gpt_cfg()).eval()
    idx = torch.randint(0, 100, (2, 8))
    out1 = m(idx)
    idx2 = idx.clone()
    idx2[:, -1] = (idx2[:, -1] + 1) % 100  # 마지막 토큰만 바꿈
    out2 = m(idx2)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
    assert not torch.allclose(out1[:, -1], out2[:, -1], atol=1e-5)


def test_text_to_gpt_end_to_end():
    ids = tokenizer.encode("Once upon a time, there was a small robot.")
    idx = torch.tensor([ids])
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, n_embd=32, n_head=4, block_size=64)
    y = GPT(cfg)(idx)
    assert y.shape == (1, len(ids), 32)


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
