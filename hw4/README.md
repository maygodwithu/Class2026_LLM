# HW4 -- TransformerBlock, GPT(decoder) 완성, 텍스트 생성

HW3에서 완성한 `MultiHeadAttention`을 **LayerNorm, FeedForward, shortcut(residual)으로 감싸서
TransformerBlock**을 만들고, 그 블록을 **여러 층 쌓아 GPT(decoder-only)를 완성**한 다음
**텍스트를 한 토큰씩 생성**해 봅니다. 교재 4장 4.2~4.7절의 형태를 따릅니다.

```
x --> [ norm1 -> MultiHeadAttention -> dropout -> + x ] --> [ norm2 -> FeedForward -> dropout -> + ] --> out   (TransformerBlock)

text --tokenizer--> ids --GPT--> logits --(마지막 위치, argmax)--> 다음 토큰 --> ids 뒤에 붙이고 반복
                     GPT = 임베딩 -> TransformerBlock x n_layer -> final_norm -> out_head
```

LayerNorm과 shortcut은 self-attention의 일부가 아니라, attention과 FeedForward를 똑같은 방식으로
감싸는 블록 구조입니다. LayerNorm(`nn.LayerNorm`)과 GELU(`nn.GELU`)는 PyTorch에 있는 것을 그대로 씁니다.

`GPT`는 HW3에서 `self.att`로 어텐션 하나를 쓰던 자리에 이제 `TransformerBlock`을 `n_layer`개 쌓고,
마지막에 `final_norm`과 `out_head`를 붙여 **어휘 크기의 점수(logits)** 를 출력합니다.

## 할 일 (5곳, 각각 한 두 줄)
각 TODO는 한 두 줄만 채우면 되고, 나머지는 이미 들어 있습니다. 바로 위아래 코드가 힌트입니다.
1. **TODO 1 (`FeedForward.__init__`, 교재 4.3)**: 4배로 늘렸던 것을 다시 `n_embd`로 줄이는 `Linear` 층 `down` 만들기
2. **TODO 2 (`TransformerBlock.forward`, 교재 4.4/4.5)**: 어텐션 블록의 끝 두 줄, dropout 적용 후 shortcut(원래 입력) 더하기
   (바로 아래에 완성된 피드 포워드 블록이 같은 모양으로 들어 있습니다)
3. **TODO 3 (`GPT.__init__`, 교재 4.6)**: 블록을 몇 층 쌓을지 `num_layers`를 정하기 (`config`에 있는 값, 쌓는 코드 `trf_blocks`는 들어 있습니다)
4. **TODO 4 (`GPT.forward`, 교재 4.6)**: `trf_blocks` 다음에 `final_norm` -> `out_head`를 통과시켜 logits 만들기
5. **TODO 5 (`GPT.generate`, 교재 4.7)**: 마지막 위치의 logits에서 가장 큰 값의 토큰 `idx_next` 고르기 (그리디 디코딩)

`MultiHeadAttention`(HW3), 블록을 쌓는 `trf_blocks` 코드, `final_norm`, `out_head`, `generate`의 반복문(문맥 자르기, 이어 붙이기)은 이미 들어 있습니다.
GELU는 교재의 식과 같은 `nn.GELU(approximate="tanh")`로 이미 들어 있습니다.

## 실행
```bash
python -m pytest test_hw4.py -v -s
```

## 기대 결과

**1) 아무것도 안 채웠을 때**: `26 failed` (실패는 모두 `NotImplementedError`)

**2) 단계별로 채워 갈 때**: 앞 단계가 되어야 뒤 단계 테스트가 돌아갑니다.

| 채운 단계 | 결과 |
|---|---|
| TODO 1 | `4 passed, 22 failed` (FeedForward 테스트 통과) |
| TODO 1, 2 | `12 passed, 14 failed` (TransformerBlock 테스트까지 통과) |
| TODO 1~3 | `15 passed, 11 failed` (GPT 구조와 파라미터 수 통과) |
| TODO 1~4 | `21 passed, 5 failed` (logits 출력 통과, 생성 테스트만 실패) |
| TODO 1~5 | `26 passed` |

**3) 모두 채웠을 때**: 아래처럼 나와야 합니다.
```
test_hw4.py::test_feedforward_structure PASSED
...
test_hw4.py::test_book_example_feedforward
출력 크기: (2, 3, 768)
PASSED
...
test_hw4.py::test_book_example_block
입력 크기: (2, 4, 768)
출력 크기: (2, 4, 768)
PASSED
...
test_hw4.py::test_book_param_count
총 파라미터 개수: 163,009,536
가중치 묶기를 고려한 파라미터 개수: 124,412,160
PASSED
...
test_hw4.py::test_book_example_logits
입력 배치:
 tensor([[6109, 3626, 6100,  345],
        [6109, 1110, 6622,  257]])
출력 크기: (2, 4, 50257)
PASSED
...
test_hw4.py::test_book_example_generate
인코딩된 ID: [15496, 11, 314, 716]
출력: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])
출력 길이: 10
출력 텍스트: Hello, I am Featureiman Byeswickattribute argue
PASSED
...
============================== 26 passed ==============================
```
- FeedForward와 TransformerBlock은 입력과 같은 크기로 출력합니다 (교재 4.3, 4.5절).
- `test_shortcut_passes_gradient`: 하위 층의 출력이 0이어도 shortcut 덕분에 입력까지 gradient가 그대로 흐릅니다 (교재 4.4절).
- GPT-2 124M 설정(`n_layer=12`, `n_embd=768`)의 파라미터 수는 163,009,536개이고, 출력층(`out_head`)을 임베딩과 공유한다고 보면 124,412,160개입니다 (교재 4.6절).
- `GPT`의 출력은 `(배치, 토큰 수, vocab_size)`입니다. 각 위치에서 "다음 토큰이 무엇일지"에 대한 점수입니다.
- 학습하기 전의 모델이라서 생성된 문장은 의미 없는 단어의 나열입니다. 이것이 정상이고, 다음 과제에서 학습합니다.
- 생성된 문장은 층을 만드는 순서가 교재와 같을 때 위와 똑같이 나옵니다 (`trf_blocks` -> `final_norm` -> `out_head`).
  테스트는 길이와 앞부분만 검사하므로 순서가 달라도 통과하지만, 문장은 달라질 수 있습니다.
- 교재 `ch04.ipynb`에 저장된 출력 문장은 위와 다릅니다. 교재 저장소의 `gpt.py`를 실행하면 위 문장이 나옵니다.

## 리포트
1. LayerNorm은 어떤 축으로 무엇을 정규화하나요? `scale`과 `shift`(PyTorch에서는 `weight`, `bias`)는 왜 필요한가요? (교재 4.2절)
2. FeedForward는 왜 `4 * n_embd`로 늘렸다가 줄이나요? attention과 달리 위치끼리 섞이지 않는다는 건 무슨 뜻인가요?
3. shortcut이 없으면 어떤 문제가 생기나요? (교재 4.4절의 gradient 예제를 참고)
4. TransformerBlock에서 `norm1`, `norm2`는 왜 각각 attention, FeedForward **앞에** 있나요?
5. `GPT`가 `(배치, 토큰 수, vocab_size)` 크기의 logits를 내는 이유는 무엇이고, `generate`는 왜 그 중 **마지막 위치**의 logits만 사용하나요?
