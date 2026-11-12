"""
학습 데이터를 만드는 파일 (교재 2.6절). 이미 완성되어 있고, 수업시간에 설명합니다. 채울 것은 없습니다.

텍스트 -> 토큰 ID -> 슬라이딩 윈도우로 (입력 x, 정답 y) 쌍 만들기 -> DataLoader 로 배치 묶기

핵심은 정답 y가 입력 x를 "한 칸 민" 것이라는 점입니다.
    x = [t0, t1, t2, t3]
    y = [t1, t2, t3, t4]
x의 각 위치에서 모델이 맞혀야 하는 것은 "바로 다음 토큰"이므로, 위치마다 하나씩 정답이 있습니다.

실행해 보기:  python data.py
"""

import os

import torch
from torch.utils.data import DataLoader, Dataset

import tokenizer

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "the-verdict.txt")


def load_text(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text, train_ratio=0.9):
    """글자 수 기준으로 앞 90%는 학습용, 뒤 10%는 검증용으로 나눕니다. (교재 5.1.2)"""
    split_idx = int(train_ratio * len(text))
    return text[:split_idx], text[split_idx:]


class GPTDataset(Dataset):  # 교재의 GPTDatasetV1

    def __init__(self, text, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 1) 텍스트 전체를 토큰 ID 리스트로 바꾼다
        token_ids = tokenizer.encode(text)

        # 2) 슬라이딩 윈도우: stride 만큼 옮겨 가며 max_length 길이로 자른다
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]  # 입력을 한 칸 민 것
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(text, batch_size, max_length, stride, shuffle=True, drop_last=True):
    """(x, y) 쌍들을 batch_size개씩 묶어 (batch_size, max_length) 크기의 배치로 내보냅니다."""
    dataset = GPTDataset(text, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


if __name__ == "__main__":
    text = load_text()
    train_text, val_text = split_text(text)
    print(f"전체 글자 수: {len(text)}  (학습 {len(train_text)} / 검증 {len(val_text)})")
    print(f"전체 토큰 수: {len(tokenizer.encode(text))}")

    max_length = 8
    loader = create_dataloader(train_text, batch_size=2, max_length=max_length,
                               stride=max_length, shuffle=False)
    x, y = next(iter(loader))
    print(f"\n첫 배치: x {tuple(x.shape)}, y {tuple(y.shape)}")
    print("x[0] :", x[0].tolist())
    print("y[0] :", y[0].tolist(), " <- x[0]을 한 칸 민 것")
    print("x[0] 텍스트:", repr(tokenizer.decode(x[0].tolist())))
    print("y[0] 텍스트:", repr(tokenizer.decode(y[0].tolist())))

    # 교재 5장 설정: max_length = stride = 256, batch_size = 2
    train_loader = create_dataloader(train_text, 2, 256, 256, shuffle=True, drop_last=True)
    val_loader = create_dataloader(val_text, 2, 256, 256, shuffle=False, drop_last=False)
    print(f"\n교재 설정 (max_length=stride=256, batch_size=2)")
    print(f"학습 배치 수: {len(train_loader)}, 검증 배치 수: {len(val_loader)}")
