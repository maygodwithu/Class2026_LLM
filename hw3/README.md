# HW3 -- FFN + Transformer Block

## 목표
HW2의 self-attention(주어짐)과, 이번에 구현할 MLP(feed-forward)를 합쳐서
"Transformer Block" 하나를 완성합니다. 이 Block을 여러 개 쌓는 것이
HW4에서 할 일입니다.

## 할 일
`block.py`에서 `MLP.forward()`와 `Block.forward()` 두 개를 구현하세요.
`LayerNorm`, `CausalSelfAttention`은 이미 완성되어 주어집니다(HW2 정답).

## 왜 residual(잔차) 연결인가?
`Block.forward()`는 `x = attn(x)`가 아니라 `x = x + attn(ln_1(x))`처럼
**더하는** 형태여야 합니다. 이 "더하기" 경로가 없으면 층을 깊게 쌓을수록
gradient가 사라지거나 폭발해서 학습이 거의 불가능해집니다.
`test_block_is_residual`은 attention과 MLP의 출력을 강제로 0으로 만들어서
`Block(x) == x`가 되는지 직접 확인하는 테스트입니다 -- 만약 residual을
빼먹었다면(overwrite했다면) 이 테스트가 실패합니다.

## 실행 / 채점
```bash
cd homework/hw3
../../.venv/bin/python -m pytest test_hw3.py -v
```

## 리포트에 적을 것
`test_block_is_residual`이 왜 "attn/mlp 출력을 0으로 만들면 Block(x)==x"라는
방식으로 residual을 검증할 수 있는지, 본인 말로 한두 문장 설명하세요.
