"""
GPT를 학습시키는 코드 (교재 5장). HW4의 model.py와 data.py(수업시간에 설명)를 사용합니다.
학습 = "다음 토큰 맞히기" 손실을 계산하고, 그 gradient로 가중치를 조금씩 고치는 일을 반복하는 것.
TODO 1: calc_loss_batch()  (한 줄)
TODO 2: perplexity()  (한 줄)
TODO 3: evaluate_model()  (한 줄)
TODO 4: train_model_simple()  (두 줄)

실행해 보기:  python train.py           (교재와 같은 124M 모델, GPU가 있으면 훨씬 빠릅니다)
              python train.py --small   (작은 모델, CPU에서도 빠르게 확인)
"""

import sys

import torch
import torch.nn.functional as F

import data
import tokenizer
from model import GPT, GPTConfig


def text_to_token_ids(text):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)  # 배치 차원 추가: (1, 토큰 수)


def token_ids_to_text(token_ids):
    flat = token_ids.squeeze(0)  # 배치 차원 제거
    return tokenizer.decode(flat.tolist())


# ---------------------------------------------------------------- 손실 (교재 5.1)
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)  # (b, t, vocab_size)
    # TODO 1: logits와 target_batch로 cross entropy 손실 loss 구하기
    #   logits (b, t, vocab_size)와 target_batch (b, t)를 각각 flatten(0, 1), flatten()으로
    #   (b*t, vocab_size)와 (b*t,)로 펼친 뒤 F.cross_entropy에 넣으세요.
    raise NotImplementedError  # 채운 뒤 이 줄은 지우세요
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """data_loader의 배치들(최대 num_batches개)에 대한 평균 손실"""
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def perplexity(loss):  # 교재 5.1.2의 팁
    """손실(텐서)을 perplexity로 바꿉니다. 모델이 매 위치에서 '평균 몇 개의 토큰 중에서 고르는 셈인지'"""
    # TODO 2: perplexity는 손실의 지수(exp)입니다.
    raise NotImplementedError  # 채운 뒤 이 줄은 지우세요


# ---------------------------------------------------------------- 평가와 샘플 생성
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # TODO 3: 평가할 때는 dropout이 꺼지도록 모델을 평가 모드로 바꾸세요.
    raise NotImplementedError  # 채운 뒤 이 줄은 지우세요
    with torch.no_grad():  # 평가에서는 gradient가 필요 없으므로 계산하지 않는다
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 학습 모드로 되돌린다
    return train_loss, val_loss


def generate_and_print_sample(model, device, start_context, max_new_tokens=50):
    model.eval()
    encoded = text_to_token_ids(start_context).to(device)
    token_ids = model.generate(encoded, max_new_tokens)  # HW4의 그리디 generate
    print(token_ids_to_text(token_ids.cpu()).replace("\n", " "))  # 한 줄로 출력
    model.train()


# ---------------------------------------------------------------- 학습 루프 (교재 5.2)
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()  # 학습 모드 (dropout 켜짐)

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 이전 배치에서 구한 gradient 지우기
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # TODO 4: loss에서 gradient를 구하고(backward), optimizer로 가중치를 갱신하기(step)
            raise NotImplementedError  # 채운 뒤 이 줄은 지우세요
            tokens_seen += input_batch.numel()
            global_step += 1

            # 일정 간격마다 학습/검증 손실을 확인
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 에포크가 끝날 때마다 지금 모델이 만드는 문장을 출력
        generate_and_print_sample(model, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def main(small=False):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 모델: 교재 124M 설정(문맥 길이만 256으로 줄임). --small 이면 작은 모델
    if small:
        config = GPTConfig(block_size=256, vocab_size=tokenizer.vocab_size,
                           n_embd=256, n_head=4, n_layer=4, dropout=0.1)
    else:
        config = GPTConfig(block_size=256, vocab_size=tokenizer.vocab_size, dropout=0.1)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # 데이터: data.py (수업시간에 설명)
    text = data.load_text()
    train_text, val_text = data.split_text(text, train_ratio=0.90)
    train_loader = data.create_dataloader(
        train_text, batch_size=2, max_length=config.block_size, stride=config.block_size,
        shuffle=True, drop_last=True)
    val_loader = data.create_dataloader(
        val_text, batch_size=2, max_length=config.block_size, stride=config.block_size,
        shuffle=False, drop_last=False)
    print(f"학습 배치 {len(train_loader)}개, 검증 배치 {len(val_loader)}개\n")

    # 학습 전/후 출력을 비교하기 위한 입력 (같은 입력을 학습 전과 후에 모델에 넣어 본다)
    prompts = ["Every effort moves you", "I HAD always thought", "He laughed again, and"]

    print("=== 학습 전 (무작위 초기화) ===")
    for p in prompts:
        generate_and_print_sample(model, device, p, max_new_tokens=30)
    print()

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=10, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you")

    print("\n=== 학습 후 (10에폭) ===")
    for p in prompts:
        generate_and_print_sample(model, device, p, max_new_tokens=30)

    print(f"\n마지막 검증 손실 {val_losses[-1]:.3f}  ->  perplexity {perplexity(torch.tensor(val_losses[-1])):.1f}")
    print(f"마지막 학습 손실 {train_losses[-1]:.3f}  ->  perplexity {perplexity(torch.tensor(train_losses[-1])):.1f}")
    return model, train_losses, val_losses, tokens_seen


if __name__ == "__main__":
    main(small="--small" in sys.argv)
