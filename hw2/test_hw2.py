import torch
import torch.nn.functional as F

import tokenizer
from model import GPT, GPTConfig, CausalAttention

D_IN, D_OUT, T = 16, 8, 6


def make(dropout=0.0):
    torch.manual_seed(0)
    m = CausalAttention(D_IN, D_OUT, T, dropout)
    x = torch.randn(2, T, D_IN)
    return m, x


# 교재 3장의 예제 입력 (Your journey starts with one step)
BOOK_INPUTS = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)

# 교재 3.5.1절에 출력된 causal attention weights (seed 789, d_in=3, d_out=2)
BOOK_CAUSAL_WEIGHTS = torch.tensor(
    [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
     [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
     [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
     [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
     [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
     [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]]
)


def book_example():
    torch.manual_seed(789)
    m = CausalAttention(3, 2, 6, 0.0)
    return m, BOOK_INPUTS.unsqueeze(0)


def hand_example():
    # W_query = W_key = W_value = 단위행렬 -> q = k = v = x (손계산용, T=3, d=2)
    m = CausalAttention(2, 2, 3, 0.0)
    with torch.no_grad():
        for W in (m.W_query, m.W_key, m.W_value):
            W.weight.copy_(torch.eye(2))
    x = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])
    return m, x


# ---------------- 1단계: self-attention (Q, K, V, attention table) ----------------

def test_output_shape():
    m, x = make()
    assert m(x).shape == (2, T, D_OUT)


def test_qkv_shape():
    m, x = make()
    m(x)
    assert m.queries.shape == m.keys.shape == m.values.shape == (2, T, D_OUT)


def test_qkv_values():
    m, x = make()
    m(x)
    assert torch.allclose(m.queries, m.W_query(x))
    assert torch.allclose(m.keys, m.W_key(x))
    assert torch.allclose(m.values, m.W_value(x))


def test_attention_table_shape():
    m, x = make()
    m(x)
    assert m.attn_scores.shape == (2, T, T)
    assert m.attn_weights.shape == (2, T, T)


def test_attention_rows_sum_to_one():
    m, x = make()
    m(x)
    assert torch.allclose(m.attn_weights.sum(dim=-1), torch.ones(2, T), atol=1e-5)


def test_last_row_matches_full_attention():
    # 마지막 위치는 causal이어도 전체를 다 보므로 mask 없는 attention과 같아야 함
    m, x = make()
    m(x)
    full = torch.softmax(m.queries @ m.keys.transpose(1, 2) / D_OUT**0.5, dim=-1)
    assert torch.allclose(m.attn_weights[:, -1], full[:, -1], atol=1e-5)


def test_book_example_last_row():
    m, x = book_example()
    m(x)
    assert torch.allclose(m.attn_weights[0, -1], BOOK_CAUSAL_WEIGHTS[-1], atol=1e-3)


def test_hand_example_q_k_v():
    m, x = hand_example()
    m(x)
    assert torch.allclose(m.queries, x) and torch.allclose(m.keys, x) and torch.allclose(m.values, x)


def test_hand_example_last_row():
    m, x = hand_example()
    m(x)
    assert torch.allclose(m.attn_weights[0, -1], torch.tensor([0.2483, 0.2483, 0.5035]), atol=1e-3)


def test_dropout_is_off_in_eval():
    m, x = make(dropout=0.5)
    m.eval()
    m(x)
    assert torch.allclose(m.attn_weights.sum(dim=-1), torch.ones(2, T), atol=1e-5)


# ---------------- 2단계: causal ----------------

def test_causal_upper_triangle_is_zero():
    m, x = make()
    m(x)
    print("\nattention table (첫 번째 문장):\n", m.attn_weights[0].detach().round(decimals=3))
    assert torch.all(torch.triu(m.attn_weights, diagonal=1) == 0), "미래 위치의 확률이 0이 아닙니다"


def test_masked_scores_are_minus_inf():
    m, x = make()
    m(x)
    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    assert torch.all(m.attn_scores[:, upper] == -torch.inf)
    assert torch.all(torch.isfinite(m.attn_scores[:, ~upper]))


def test_first_position_sees_only_itself():
    m, x = make()
    m(x)
    assert torch.allclose(m.attn_weights[:, 0, 0], torch.ones(2))


def test_zero_after_own_position():
    m, x = make()
    m(x)
    for i in range(T):
        assert torch.all(m.attn_weights[:, i, i + 1:] == 0)
    assert torch.allclose(m.attn_weights.sum(dim=-1), torch.ones(2, T), atol=1e-5)


def test_hand_example_causal_table():
    m, x = hand_example()
    m(x)
    print("\nattention table (손계산 예제):\n", m.attn_weights[0].detach().round(decimals=3))
    expected = torch.tensor([[1.0, 0.0, 0.0],
                             [0.3302, 0.6698, 0.0],
                             [0.2483, 0.2483, 0.5035]])
    assert torch.allclose(m.attn_weights[0], expected, atol=1e-3)


def test_book_example_causal_table():
    m, x = book_example()
    m(x)
    print("\nattention table (교재 3.5.1절 예제):\n", m.attn_weights[0].detach().round(decimals=4))
    assert torch.allclose(m.attn_weights[0], BOOK_CAUSAL_WEIGHTS, atol=1e-3)


def test_future_token_does_not_change_past():
    m, x = make()
    out1 = m(x)
    x2 = x.clone()
    x2[:, -1] = torch.randn(2, D_IN)
    out2 = m(x2)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)


def test_matches_torch_causal_attention():
    m, x = make()
    m(x)
    ref = F.scaled_dot_product_attention(m.queries, m.keys, m.values, is_causal=True)
    assert torch.allclose(m.attn_weights @ m.values, ref, atol=1e-5)


# ---------------- HW1과 이어보기: text -> tokenizer -> GPT -> attention ----------------

def test_text_to_attention_end_to_end():
    ids = tokenizer.encode("Once upon a time, there was a small robot.")
    idx = torch.tensor([ids])
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, n_embd=32, block_size=64)
    x = GPT(cfg)(idx)
    y = CausalAttention(32, 32, cfg.block_size, 0.0)(x)
    assert y.shape == (1, len(ids), 32)


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
