from src.models.scratch import BaselineCNN
from src.models.adapter import AdapterCNN
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

    os.makedirs(RESULT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loader(batch_size=64)
    
    baseline_ckpt = os.path.join(RESULT_DIR, "cnn_best_val_loss.pth")

    # baseline weighㅅ복사
    # 1) baseline 모델 로드 (이미 학습 완료)
    baseline_model = BaselineCNN(num_classes=10).to(device)
    baseline_model.load_state_dict(torch.load(baseline_ckpt, map_location=device))
    baseline_model.eval()

    # 2) Adapter 기반 CNN 생성
    adapter_model = AdapterCNN(
        num_classes=10,
        bottleneck_ratio=2,
        dropout=0.1
    ).to(device)

    # 3) conv1 복사
    adapter_model.conv1.load_state_dict(
        baseline_model.conv1[0].state_dict()
    )

    # 4) conv2, conv3 → 각각 Adapter의 base_conv로 복사
    adapter_model.conv2_adapter.base_conv.load_state_dict(
        baseline_model.conv2[0].state_dict()
    )

    adapter_model.conv3_adapter.base_conv.load_state_dict(
        baseline_model.conv3[0].state_dict()
    )

    # 5) fc1, fc_out 복사
    adapter_model.fc1.load_state_dict(baseline_model.fc1.state_dict())
    adapter_model.fc_out.load_state_dict(baseline_model.fc_out.state_dict())

    # freeze + optimizer 세팅

    # 1) 전체 freeze
    for p in adapter_model.parameters():
        p.requires_grad = False

    # 2) Adapter branch (down/up)만 trainable
    for m in [adapter_model.conv2_adapter, adapter_model.conv3_adapter]:
        for p in m.down.parameters():
            p.requires_grad = True
        for p in m.up.parameters():
            p.requires_grad = True

    # 3) FC도 trainable
    for p in adapter_model.fc1.parameters():
        p.requires_grad = True
    for p in adapter_model.fc_out.parameters():
        p.requires_grad = True

    # 4) Optimizer: trainable 파라미터만
    adapter_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, adapter_model.parameters()),
        lr=2e-4,
        weight_decay=1e-5 #기존:1e-5
    )
    criterion = nn.CrossEntropyLoss()

    adapter_ckpt_path = os.path.join(RESULT_DIR, "adapter_cnn_best_val_loss.pth")

    history_adapter = train_model(
        model=adapter_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=adapter_optimizer,
        criterion=criterion,
        epochs=30,
        ckpt_path=adapter_ckpt_path,
        model_name="Adapter",
        device = device,
    )
    
    history_path = os.path.join(RESULT_DIR, "history_adapter.json")
    with open(history_path, "w") as f:
        json.dump(history_adapter, f, indent=2)


    adapter_test_loss, adapter_test_acc = eval_model(
        model=adapter_model,
        test_loader=test_loader,
        criterion=criterion,
        ckpt_path=adapter_ckpt_path,
        model_name="Adapter",
        device = device,
    )
    
    test_result_path = os.path.join(RESULT_DIR, "test_adapter.json")
    with open(test_result_path, "w") as f:
        json.dump(
            {
                "test_loss": adapter_test_loss,
                "test_acc": adapter_test_acc,
            },
            f,
            indent=2,
        )

if __name__ == "__main__":
    main()

