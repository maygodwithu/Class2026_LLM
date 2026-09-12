# GPT-2 사전학습 BPE 토크나이저 (nanoGPT가 실제로 쓰는 방식). 구현할 것 없음.

import tiktoken

_enc = tiktoken.get_encoding("gpt2")

vocab_size = _enc.n_vocab
eot_token = _enc.eot_token


def encode(text: str) -> list[int]:
    return _enc.encode_ordinary(text)


def encode_with_eot(text: str) -> list[int]:
    return _enc.encode_ordinary(text) + [eot_token]


def decode(ids: list[int]) -> str:
    return _enc.decode(ids)


if __name__ == "__main__":
    sample = "Once upon a time, there was a small robot."
    ids = encode(sample)
    print(f"text:  {sample!r}")
    print(f"ids:   {ids}")
    print(f"decode(ids) == text: {decode(ids) == sample}")
