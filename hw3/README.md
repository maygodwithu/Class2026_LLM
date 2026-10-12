# HW3 -- Multi-Head Attention (Self-Attention 마무리)

HW2의 `CausalAttention`(head 1개)이 **MultiHeadAttention(head 여러 개)으로 진화**합니다.
`model.py`에는 HW2의 `CausalAttention` 선언부 한 줄이 주석으로 남아 있고, 그 아래에 `MultiHeadAttention`이 있습니다.
교재 3장 3.6절의 형태를 따르며, 이번 과제로 **self-attention이 완성**됩니다.

```
text --tokenizer--> ids --GPT(임베딩 HW1 + 어텐션 HW3)--> context_vec
```

`GPT`가 이제 `self.att`로 `MultiHeadAttention`을 사용합니다 (임베딩 -> 어텐션).

LayerNorm, residual(shortcut), FFN은 self-attention 바깥을 감싸는 구조라서 다음 과제에서
TransformerBlock으로 다룹니다. 교재 3.6.1절(`CausalAttention`을 여러 개 쌓는 방식)은 읽어만 보세요.

## 할 일 (2단계)
`MultiHeadAttention.forward()`의 두 곳을 채우세요. (교재 3.6.2)
1. **TODO 1 (head 나누기)**: `view`와 `transpose`로 `(b, num_tokens, d_out)` -> `(b, num_heads, num_tokens, head_dim)`
2. **TODO 2 (head 합치기)**: `(b, num_heads, num_tokens, head_dim)` -> `(b, num_tokens, d_out)`, 그 다음 `out_proj`

## 실행
```bash
python -m pytest test_hw3.py -v -s
```
`-s`를 붙이면 head별 attention table이 출력됩니다.

## 기대 결과

**1) 아무것도 안 채웠을 때**: `1 passed, 15 failed` (실패는 모두 `NotImplementedError`)

**2) TODO 1만 채웠을 때**: head 나누기와 attention table 테스트는 통과하고, 합치기가 필요한 8개만 실패합니다.
(`8 passed, 8 failed`)

**3) TODO 1, 2를 모두 채웠을 때**: 아래처럼 나와야 합니다.
```
test_hw3.py::test_split_shapes PASSED
test_hw3.py::test_split_values PASSED
...
test_hw3.py::test_causal_upper_triangle_is_zero_for_every_head
attention table (첫 번째 문장, head 0):
 tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.4690, 0.5310, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1670, 0.4790, 0.3540, 0.0000, 0.0000, 0.0000],
        [0.2750, 0.2520, 0.2360, 0.2380, 0.0000, 0.0000],
        [0.2050, 0.2120, 0.2620, 0.1420, 0.1790, 0.0000],
        [0.1930, 0.1240, 0.1240, 0.2410, 0.1180, 0.2000]])

attention table (첫 번째 문장, head 1):
 tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.6890, 0.3110, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.4100, 0.2900, 0.3000, 0.0000, 0.0000, 0.0000],
        [0.2750, 0.2860, 0.2740, 0.1650, 0.0000, 0.0000],
        [0.2210, 0.1910, 0.2100, 0.1580, 0.2190, 0.0000],
        [0.1930, 0.1430, 0.1700, 0.1580, 0.1660, 0.1700]])
PASSED
...
test_hw3.py::test_book_example_multihead
context_vecs.shape: (2, 6, 2)
PASSED
...
============================== 16 passed ==============================
```
- head마다 attention table이 따로 있고, 두 head의 값은 서로 다릅니다 (둘 다 오른쪽 위는 0).
- 교재 3.6.2절의 예제 출력(크기 `(2, 6, 2)`)과 같은 값이 나오는지도 비교합니다.
- head별 table은 랜덤 입력이라 torch 버전에 따라 숫자가 조금 다를 수 있지만 모양은 같아야 합니다.

## 리포트
1. 교재 3.6.1(`CausalAttention`을 여러 개 쌓기)과 3.6.2(이번 구현)는 출력 크기와 계산 방식이 어떻게 다른가요?
2. `view`로 나눈 뒤 `transpose(1, 2)`를 하는 이유는? (head 차원이 어디로 가는지)
3. softmax 전에 `d_out`이 아니라 `keys.shape[-1]`(= `head_dim`)의 제곱근으로 나누는 이유는?
4. `out_proj`는 무슨 역할을 하나요?
