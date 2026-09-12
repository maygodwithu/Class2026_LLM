# HW4 -- Full Model Assembly + Decoding

## 목표
HW1(입력 임베딩), HW2(attention), HW3(MLP+Block)를 전부 조립해서 완전한
GPT를 만들고, 다음 토큰을 실제로 "생성"하는 부분까지 구현합니다. 이번
과제가 끝나면 모델 구조 자체는 완성입니다 -- 다음 과제(HW5)부터는 이
모델을 실제로 학습시킵니다.

## 할 일
`model.py`에서 세 부분을 구현하세요:
1. `GPT.__init__` -- wte/wpe/blocks/ln_f/lm_head를 만들고 **weight tying**
2. `GPT.forward` -- 임베딩 → N개 블록 → 최종 LayerNorm → lm_head → (logits, loss)
3. `GPT.generate` -- top-k/temperature를 적용한 autoregressive 샘플링

## Weight tying이란?
`self.lm_head.weight = self.wte.weight`로 출력 헤드가 입력 임베딩 행렬을
그대로 재사용하게 만드는 것입니다. `test_weight_tying`은 단순히 값이
같은지가 아니라 **같은 텐서 객체인지**(`data_ptr()` 비교)를 확인합니다.
왜 중요한가: 이 정도 모델 크기에서는 어휘(vocab) 임베딩 하나가 전체
파라미터의 20~35%를 차지할 수 있습니다 (`reference/configs.py` 참고) --
tying으로 그 비용을 거의 절반으로 줄이는 셈입니다.

## 실행 / 채점
```bash
cd homework/hw4
../../.venv/bin/python -m pytest test_hw4.py -v
```

`test_loss_near_random_init_baseline`이 특히 중요합니다: 학습 전 모델의
loss는 `ln(vocab_size)`에 가까워야 합니다 -- 이는 우리가 실제 서버에서
`tiny_1m`(vocab=4000)을 학습시켰을 때 시작 loss가 8.31, `ln(4000)=8.29`로
거의 정확히 일치했던 것과 같은 원리입니다 (모델이 균등 확률로 찍는
상태).

## 리포트에 적을 것
1. `test_loss_near_random_init_baseline`이 통과한다는 게 왜 "모델이
   제대로 조립됐다"는 증거가 될 수 있는지 설명하세요.
2. `top_k`를 크게(예: vocab_size 전체) 주는 것과 작게(예: 5) 주는 것이
   생성 결과에 어떤 영향을 줄지 짧게 추측해보세요.
