# HW5: GPT 학습하기 (사전학습)

HW4에서 완성한 GPT(decoder-only)는 아직 학습하지 않아서 의미 없는 단어만 내놓습니다.
이번 과제에서는 **텍스트로 이 모델을 학습시키는 코드**를 완성합니다. (교재 5장)

## 파일 구성

| 파일 | 설명 |
|---|---|
| `model.py` | HW4에서 완성한 GPT. **고치지 않고 가져다 씁니다.** |
| `data.py` | 학습 데이터 만들기 (교재 2.6절). 이미 완성되어 있고, 수업시간에 설명합니다. |
| `train.py` | 손실 계산, 평가, 학습 루프. **여기에서 네 곳을 채웁니다.** |
| `the-verdict.txt` | 학습에 쓰는 짧은 소설 (글자 20,479개, 토큰 5,145개) |
| `tokenizer.py` | 이전 과제와 같은 토크나이저 |
| `test_hw5.py` | 테스트 |

`python data.py`를 실행하면 텍스트가 (입력 x, 정답 y)로 어떻게 잘리는지 볼 수 있습니다. 정답 y는 입력 x를 **한 칸 민 것**입니다.

## 이번 과제에서 할 일

`train.py`에서 네 곳을 채우세요. 각각 **한 두 줄**입니다.

1. **TODO 1 (`calc_loss_batch`)**: logits와 정답으로 cross entropy 손실 `loss`를 구합니다. logits `(b, t, vocab_size)`와 정답 `(b, t)`를 `flatten`으로 펼쳐서 `F.cross_entropy`에 넣습니다. (교재 5.1절)
2. **TODO 2 (`perplexity`)**: 손실을 perplexity로 바꿉니다. (교재 5.1절)
3. **TODO 3 (`evaluate_model`)**: 평가할 때는 dropout이 꺼지도록 모델을 평가 모드로 바꿉니다.
4. **TODO 4 (`train_model_simple`)**: 손실에서 gradient를 구하고(backward), optimizer로 가중치를 갱신합니다(step). (교재 5.2절)

앞 단계를 채워야 뒤 단계 테스트가 돌아갑니다. 단계별 **정상** 결과는 다음과 같습니다.

```
아무것도 안 채움 : 15 failed, 7 passed   (실패는 모두 NotImplementedError)
TODO 1           : 10 failed, 12 passed
TODO 1~2         :  7 failed, 15 passed
TODO 1~3         :  5 failed, 17 passed
TODO 1~4         : 22 passed
```

안 채웠을 때 통과하는 7개는 `data.py`와 `generate_and_print_sample`을 확인하는 테스트입니다.

## 실행

반드시 `hw5` 폴더 안에서 실행하세요.

```bash
python -m pytest test_hw5.py -v -s
```

모두 통과하면 실제로 학습해 봅니다.

```bash
python train.py --small     # 작은 모델: CPU에서 약 20초, 메모리 약 1.4GB
python train.py             # 교재의 124M 모델: CPU에서 약 1분(서버 기준, 개인 PC는 더 걸림), 메모리 약 4.3GB
```

- GPU가 있으면 자동으로 사용합니다.
- 124M 모델은 메모리를 많이 쓰니, PC 메모리가 8GB 이하이면 `--small`을 쓰세요. Google Colab에서는 둘 다 실행됩니다.
- 필요한 패키지: `pip install torch tiktoken pytest`

## 기대 결과

### 테스트 `-s` 출력
- 학습 전 124M 모델의 손실: **train 10.9876, val 10.9811** (교재 5.1절과 같은 값). 이는 어휘 수 50,257개 중에서 무작위로 고르는 수준(`ln(50257) = 10.82`)이고, perplexity는 약 59,000입니다.
- 작은 모델을 3에폭 학습하면 손실이 약 10.9에서 5.7로 내려갑니다.
- 배치 하나를 80번 보여 주면 손실이 약 10.9에서 0.1 아래로 내려갑니다(외우기).

### `python train.py` 출력 (10에폭, 90스텝)

```
Ep 1 (Step 000000): Train loss 9.8 안팎, Val loss 9.9 안팎
...
Ep 10 (Step 000085): Train loss 0.5~1.0, Val loss 6.3 안팎
```

- 학습 손실은 0.5~1.0까지 내려가는데, 검증 손실은 6.1~6.4에서 멈춥니다. 데이터가 작아서 모델이 소설을 **외우기** 때문입니다. (과적합)
- 소수점 아래 값은 CPU/GPU 등 실행 환경에 따라 조금씩 다릅니다. 교재 값(Step 0: 9.818 / 9.928, 마지막: 0.901 / 6.328)과 같은 흐름이면 정상입니다.
- 에포크마다 `Every effort moves you`에 이어지는 문장이 출력되어, 학습이 진행되면서 문장이 어떻게 바뀌는지 볼 수 있습니다.

### 학습 전/후 출력 비교

`python train.py`는 학습을 시작하기 전과 끝난 뒤에 **같은 입력 세 개**로 문장을 만들어 보여 줍니다.

```
=== 학습 전 (무작위 초기화) ===
Every effort moves you rentingetic wasn... Mortgage TT remember gard ACTIONSussed...
I HAD always thought Plex open financing Superman communities Observatory...
He laughed again, and Driverackersounded seminLab interruption Foster...

=== 학습 후 (10에폭) ===
Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!" ...
I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the ...
He laughed again, and Jack Gisburn rather a cheap genius--...
```

- **학습 전**: 의미 없는 단어의 나열입니다. 이 출력은 시드를 고정했기 때문에 실행 환경이 달라도 같습니다.
- **학습 후**: 소설에 나오는 말투와 문장이 나옵니다. 입력에 따라서는 소설 속 문장을 **그대로 다시 써 냅니다**. 학습 데이터가 작아서 모델이 소설을 외웠기 때문입니다. 학습 후 문장은 CPU/GPU 등 환경에 따라 조금씩 다를 수 있습니다.

## 참고

- `model.py`와 `data.py`는 수정하지 않습니다.
- 124M 모델에서 문장 내용은 환경에 따라 달라질 수 있으니, 정확히 같은 문장이 나오는지가 아니라 **손실이 줄어드는지**를 확인하세요.

## 제출물 (eClass 업로드)

1. 완성한 `train.py` (코드 파일 그대로 제출)
2. **작성한 코드의 역할 설명**을 코드가 아니라 **줄글로** 적어 제출합니다. 다음 내용을 포함해 주세요.
   - 정답 y가 입력 x를 한 칸 민 것인 이유, 그리고 한 번의 `model(input_batch)`로 256개 위치의 정답이 한꺼번에 생기는 이유 (HW2의 causal mask와 연결)
   - `logits.flatten(0, 1)`로 펼치는 이유, 그리고 학습 전 손실이 약 10.8(`ln(vocab_size)`)인 이유. perplexity는 무엇을 뜻하는지
   - 평가할 때 `model.eval()`과 `torch.no_grad()`가 각각 하는 일
   - `zero_grad()`, `backward()`, `step()`이 각각 하는 일
   - 학습 손실은 계속 내려가는데 검증 손실은 6 근처에서 멈추는 이유, 그리고 해결하려면 무엇이 필요한지
   - `python train.py`가 출력하는 **학습 전과 학습 후의 문장**을 비교해서, 무엇이 어떻게 달라졌는지 (직접 실행한 결과를 근거로 쓰세요)

(`test_hw5.py` 실행 결과가 통과했다는 것만으로는 이해도를 판단할 수 없으니, 반드시 본인 말로 설명을 적어 제출합니다.)
