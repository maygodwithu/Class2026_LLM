# HW1 -- Input Representation

`model.py`는 진짜 nanoGPT 코드입니다 (karpathy/nanoGPT, MIT). 이번 학기
동안 이 파일을 계속 채워서 완전한 nanoGPT로 완성해 나갑니다.

```
text --tokenizer.py(given)--> token ids --model.py(GPT)--> vectors
```

## 할 일
`model.py`의 `GPT.forward()`에서 마지막 한 줄만 채우면 됩니다: 위치
임베딩을 찾아서 `tok_emb`에 더하고, dropout 적용해서 반환하세요.

## 실행
```bash
python -m pytest test_hw1.py -v -s
```
7개 테스트 통과하면 완료. `-s`를 붙이면 `test_tokenizer_splits_a_word_into_multiple_tokens`가
실제 토큰 id를 출력해줍니다 -- 단어 하나가 여러 토큰으로 쪼개지는 걸
직접 확인하세요.

## 리포트
1. `tokenizer.py`를 보고 nanoGPT가 왜 토크나이저는 새로 학습하지 않고 GPT-2 것을 그대로 쓰는지 설명하세요.
2. 위치 임베딩을 빼면(`+ pos_emb` 제거) 어떤 테스트가 왜 실패하나요?
3. `model.py`에 지금 없는 `CausalSelfAttention`/`MLP`/`Block`은 각각 어느 과제에서 채워질까요?
