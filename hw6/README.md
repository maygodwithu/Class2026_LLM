# HW6 -- Fine-tuning

## 목표
HW5에서 TinyStories 전체로 학습시킨 모델을, 특정 도메인(예: "용이 나오는
이야기")에 맞춰 추가로 학습시킵니다. 이번 과제의 핵심은 "무엇을 학습시킬
것인가"를 명시적으로 결정하는 것입니다 -- 모델 전체를 다시 학습시키는 게
아니라, 일부 파라미터는 얼리고(freeze) 일부만 업데이트합니다.

`model.py`, `configs.py`, `data/`, 그리고 `get_batch`/`get_lr`/`train_step`
(HW5에서 이미 검증됨)은 전부 주어집니다. 이번 과제는 오직 **어떤
파라미터를 학습시킬지 고르는 것**에 관한 것입니다.

## 할 일
`finetune.py`에서 2개 함수를 구현하세요:
1. **`freeze_backbone(model, n_finetune_layers)`** -- 마지막
   `n_finetune_layers`개 블록만 남기고 나머지 블록을 얼립니다
   (`requires_grad = False`).
2. **`configure_finetune_optimizer(model, cfg)`** -- HW5의 옵티마이저와
   똑같지만, 얼린 파라미터는 옵티마이저에 아예 넘기지 않습니다.

## 왜 임베딩(`wte`)은 얼리지 않는가?
`model.py`를 보면 `lm_head.weight = wte.weight`로 **같은 텐서**입니다
(weight tying, HW4에서 다룸). 만약 `wte`를 얼리면 `lm_head`도 자동으로
얼어버립니다 -- 둘이 같은 객체이기 때문입니다. 그래서 이번 과제에서는
임베딩과 출력 헤드는 항상 학습 가능하게 두고, **중간 트랜스포머 블록만
선택적으로 얼리는** 방식을 씁니다.

## 실행 / 채점 (1단계: 유닛테스트)
```bash
cd homework/hw6
../../.venv/bin/python -m pytest test_hw6.py -v
```
특히 `test_finetune_updates_only_unfrozen_params`가 중요합니다 -- 실제로
몇 스텝 학습을 돌려보고, 얼린 블록의 파라미터 값이 **단 하나도 바뀌지
않았는지** 직접 확인합니다.

## 실행 / 채점 (2단계: 진짜 fine-tuning)
HW5에서 학습시킨 `tiny_1m` 체크포인트가 있어야 합니다
(`../hw5/out/tiny_1m/ckpt.pt` 또는 본인이 학습시킨 경로).

```bash
cd homework/hw6

# 1) "용" 키워드가 들어간 이야기만 추려서 fine-tuning용 말뭉치 생성
../../.venv/bin/python filter_stories.py \
    --input ../hw5/data_cache/raw/TinyStoriesV2-GPT4-valid.txt \
    --keyword dragon --output data_cache/raw/dragon_stories.txt

# 2) 주의: 반드시 --reuse-tokenizer로 base 모델과 같은 토크나이저를 재사용!
../../.venv/bin/python -m data.prepare \
    --input data_cache/raw/dragon_stories.txt \
    --tokenizer bpe --out-dir data_cache/dragon \
    --reuse-tokenizer ../hw5/data_cache/tiny_1m/tokenizer_bpe.json

# 3) fine-tuning 실행 (GPU 0)
CUDA_VISIBLE_DEVICES=0 ../../.venv/bin/python finetune.py \
    --base-ckpt ../hw5/out/tiny_1m/ckpt.pt \
    --data-dir data_cache/dragon \
    --n-finetune-layers 1 --max-steps 300
```

fine-tuning 전/후 생성 샘플을 비교해보세요 (base 체크포인트와 fine-tuned
체크포인트 각각에 `../hw5`의 `sample.py`류 스크립트를 써서, 혹은 직접
`model.generate()`를 호출해서).

## 리포트에 적을 것
1. fine-tuning 전/후 생성 샘플을 나란히 놓고, 도메인(키워드)에 맞게
   달라진 부분을 짚어보세요.
2. `n_finetune_layers`를 1에서 전체 레이어 수로 바꿔서(=전체 fine-tuning)
   다시 돌려보고, val loss와 생성 품질이 어떻게 달라지는지 비교하세요.
3. `--reuse-tokenizer`를 빼고(=새 토크나이저를 학습해서) fine-tuning을
   돌리면 어떤 문제가 생길지 설명하세요 (직접 돌려봐도 되고, 원리로
   추론해도 됩니다).
