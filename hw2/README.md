# HW2 -- Self-Attention

`model.py`는 HW1의 `GPT`(입력 표현, 정답 포함)에 `CausalAttention`(head 1개)을 이어붙인 파일입니다.
`CausalAttention`은 교재 3장(3.4 셀프 어텐션, 3.5 코잘 어텐션)의 형태를 따릅니다.

```
text --tokenizer--> ids --GPT(HW1)--> x --CausalAttention(HW2)--> context_vec
```

## 할 일
`CausalAttention.forward()`의 두 단계를 채우세요.
1. **TODO 1 (self-attention)**: `keys`, `queries`, `values`를 만들고 `attn_scores = queries @ keys^T` 계산
2. **TODO 2 (causal)**: mask가 1인 위치(미래)의 `attn_scores`를 `-inf`로 채우기

## 실행
```bash
python -m pytest test_hw2.py -v -s
```
`-s`를 붙이면 attention table이 출력됩니다.

## 기대 결과

**1) 아무것도 안 채웠을 때**: 19개 모두 `NotImplementedError`로 실패합니다.

**2) TODO 1만 채웠을 때**: 1단계 테스트는 통과하고 causal 테스트 8개는 실패하는 게 정상입니다.
```
test_hw2.py::test_output_shape PASSED
test_hw2.py::test_qkv_shape PASSED
test_hw2.py::test_qkv_values PASSED
test_hw2.py::test_attention_table_shape PASSED
...
test_hw2.py::test_causal_upper_triangle_is_zero FAILED
test_hw2.py::test_masked_scores_are_minus_inf FAILED
...
========================= 8 failed, 11 passed =========================
```

**3) TODO 1, 2를 모두 채웠을 때**: 아래처럼 나와야 합니다.
```
test_hw2.py::test_output_shape PASSED
test_hw2.py::test_qkv_shape PASSED
...
test_hw2.py::test_causal_upper_triangle_is_zero
attention table (첫 번째 문장):
 tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.4120, 0.5880, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2970, 0.4600, 0.2430, 0.0000, 0.0000, 0.0000],
        [0.4490, 0.1580, 0.2110, 0.1820, 0.0000, 0.0000],
        [0.2980, 0.1610, 0.1280, 0.2270, 0.1860, 0.0000],
        [0.1230, 0.2040, 0.1690, 0.1940, 0.1600, 0.1500]])
PASSED
...
test_hw2.py::test_hand_example_causal_table
attention table (손계산 예제):
 tensor([[1.0000, 0.0000, 0.0000],
        [0.3300, 0.6700, 0.0000],
        [0.2480, 0.2480, 0.5030]])
PASSED
test_hw2.py::test_book_example_causal_table
attention table (교재 3.5.1절 예제):
 tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]])
PASSED
...
============================== 19 passed ==============================
```
- 세 표 모두 **오른쪽 위가 0인 삼각형**이고 각 행의 합이 1입니다.
- "손계산 예제"와 "교재 3.5.1절 예제" 표는 위 값과 같아야 합니다 (교재 표와 대조해 보세요).
- 첫 번째 표는 랜덤 입력이라 torch 버전에 따라 숫자가 조금 다를 수 있지만, 삼각형 모양은 같아야 합니다.

## 리포트
1. Q, K, V는 각각 어떤 역할을 하나요?
2. attention table의 한 행의 합이 1인 이유는? 그 행은 무엇을 뜻하나요?
3. 미래 위치를 0이 아니라 `-inf`로 채운 뒤 softmax를 하는 이유는? (교재 3.5.1절의 "곱하고 다시 정규화"하는 방식과 비교)
