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
1단계만 채우면 "1단계" 테스트는 통과하고 "2단계(causal)" 테스트는 실패하는 게 정상입니다.
`-s`를 붙이면 attention table이 출력됩니다 (causal이면 오른쪽 위가 0인 삼각형).
교재 3.5.1절에 나온 예제 표와도 비교합니다.

## 리포트
1. Q, K, V는 각각 어떤 역할을 하나요?
2. attention table의 한 행의 합이 1인 이유는? 그 행은 무엇을 뜻하나요?
3. 미래 위치를 0이 아니라 `-inf`로 채운 뒤 softmax를 하는 이유는? (교재 3.5.1절의 "곱하고 다시 정규화"하는 방식과 비교)
