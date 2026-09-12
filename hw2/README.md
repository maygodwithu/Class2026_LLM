# HW2 -- Causal Self-Attention

## 목표
트랜스포머의 핵심 연산인 **causal multi-head self-attention**을 구현합니다.
각 위치가 자기 자신과 그 이전 위치들의 정보를 (가중합으로) 모을 수 있게
하되, 미래 위치는 절대 볼 수 없게 만드는 게 핵심입니다.

## 할 일
`attention.py`의 `CausalSelfAttention.forward()`를 구현하세요. HW1에서
만든 입력 표현 `x: (B, T, C)`를 받아서 같은 shape을 반환합니다.

## 왜 causal(인과적)이어야 하는가?
언어모델은 "지금까지 나온 단어들로 다음 단어를 예측"하는 모델입니다.
학습 시점에 미래 토큰을 미리 볼 수 있다면 그건 "예측"이 아니라 "정답을
컨닝"하는 것과 같습니다. `test_causal_masking`이 바로 이걸 검증합니다:
마지막 토큰의 값을 바꿔도 그 이전 위치들의 출력은 전혀 달라지지 않아야
합니다.

## 실행 / 채점
```bash
cd homework/hw2
../../.venv/bin/python -m pytest test_hw2.py -v
```

## 리포트에 적을 것
1. `is_causal=True`를 `is_causal=False`로 바꾸면 어떤 테스트가 실패하나요?
2. `n_head`를 1로 바꿨을 때와 4로 바꿨을 때, 모델이 표현할 수 있는 것에
   어떤 차이가 있을지 짧게 추측해서 적어보세요 (정답은 다음 과제에서
   실제로 비교해봅니다).
