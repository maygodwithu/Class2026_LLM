# HW5 -- The Training Loop

## 목표
지금까지(HW1-4) 완성한 모델은 랜덤 초기화 상태였습니다. 이번 과제에서는
실제로 TinyStories 데이터를 보여주면서 그 모델을 학습시키는 데 필요한
네 가지를 구현합니다.

`model.py`, `configs.py`, `data/`는 전부 완성된 상태로 주어집니다 (직접
짠 모델과 다를 수 있지만, 모두가 같은 정답 모델로 학습해야 이번 과제의
채점 기준이 공정해지기 때문입니다).

## 할 일
`train.py`에서 4개 함수를 구현하세요:
1. **`get_batch`** -- 데이터에서 랜덤한 위치를 골라 (입력, 정답) 쌍을 만듭니다. 정답은 입력을 한 칸 밀어놓은 것뿐입니다 ("다음 토큰 맞히기").
2. **`get_lr`** -- warmup(서서히 증가) + cosine decay(서서히 감소) 학습률 스케줄.
3. **`configure_optimizer`** -- AdamW, 단 weight decay는 2차원 이상 파라미터에만.
4. **`train_step`** -- forward → backward → gradient clipping → optimizer step, 한 스텝.

## 실행 / 채점 (1단계: 로컬 유닛테스트, CPU만 있으면 됨)
```bash
cd homework/hw5
../../.venv/bin/python -m pytest test_hw5.py -v
```
작은 합성 데이터로 실제로 loss가 줄어드는지까지 확인하는 통합 테스트
(`test_train_step_reduces_loss_over_many_steps`)가 포함되어 있어서, GPU나
진짜 데이터 없이도 몇 초 안에 4개 함수가 전부 맞는지 확인할 수 있습니다.

## 실행 / 채점 (2단계: 진짜 TinyStories로 서버 GPU에서 학습)
유닛테스트를 통과했다면, 실제 데이터로 진짜 학습을 돌려보세요:
```bash
cd homework/hw5

# 1) 데이터 다운로드 (22.5MB, 최초 1회)
../../.venv/bin/python -m data.download

# 2) 토크나이저 학습 + train.bin/val.bin 생성
../../.venv/bin/python -m data.prepare \
    --input data_cache/raw/TinyStoriesV2-GPT4-valid.txt \
    --tokenizer bpe --vocab-size 4000 --out-dir data_cache/tiny_1m

# 3) GPU 0에서 실제 학습 (약 500 스텝, 수 초~수십 초)
CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python train.py \
    --config tiny_1m --data-dir data_cache/tiny_1m
```
step 0의 loss가 `ln(vocab_size)`(vocab=4000이면 약 8.29) 근처에서 시작해서,
스텝이 진행될수록 꾸준히 감소하는지 확인하세요. (기준 실행에서는 500
스텝 만에 8.31 -> 4.33까지 내려갔습니다.)

## 리포트에 적을 것
1. `get_batch`에서 왜 정답(`y`)이 입력(`x`)을 "한 칸 민 것"이어야 하는지 설명하세요.
2. 실제 학습 loss 곡선(step vs loss)을 캡처하거나 표로 정리해서 제출하세요.
3. `warmup_steps`를 0으로 바꿔서 돌려보고, 초반 loss 곡선이 어떻게 달라지는지 비교하세요.
