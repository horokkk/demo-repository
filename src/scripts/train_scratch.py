from src.models.scratch import BaselineCNN
from src.utils.train import train_model
from src.utils.eval import eval_model
from src.data.process import get_data_loader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json

RESULT_DIR = "results"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_data_loader(batch_size=64)

    # 스크래치용 baseline 모델: 새로 만들고, 가중치 로드 X
    baseline_model = BaselineCNN(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(baseline_model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 체크포인트 경로
    baseline_ckpt = os.path.join(RESULT_DIR, "cnn_best_val_loss.pth")

    # train_model / eval_model 은 이전에 정의해둔 그 버전 사용
    history_base = train_model(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=30,
        ckpt_path=baseline_ckpt,
        model_name="Scratch",
        device = device,
    )

    history_path = os.path.join(RESULT_DIR, "history_scratch.json")
    with open(history_path, "w") as f:
        json.dump(history_base, f, indent=2)



    base_test_loss, base_test_acc = eval_model(
        model=baseline_model,
        test_loader=test_loader,
        criterion=criterion,
        ckpt_path=baseline_ckpt,
        model_name="Scratch"
    )
    test_result_path = os.path.join(RESULT_DIR, "test_scratch.json")
    with open(test_result_path, "w") as f:
        json.dump(
            {
                "test_loss": base_test_loss,
                "test_acc": base_test_acc,
            },
            f,
            indent=2,
        )

if __name__ == "__main__":
    main()