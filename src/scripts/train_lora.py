from src.models.scratch import BaselineCNN
from src.models.lora import LoRACNN
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


    # 2) LoRA 모델 생성
    lora_model = LoRACNN(num_classes=10, r=8, lora_alpha=16, lora_dropout=0.4).to(device)

    # 3) baseline weight 복사
    lora_model.conv1.load_state_dict(
        baseline_model.conv1[0].state_dict()
    )

    lora_model.conv2_lora.base_conv.load_state_dict(
        baseline_model.conv2[0].state_dict()
    )

    lora_model.conv3_lora.base_conv.load_state_dict(
        baseline_model.conv3[0].state_dict()
    )

    lora_model.fc1.load_state_dict(baseline_model.fc1.state_dict())
    lora_model.fc_out.load_state_dict(baseline_model.fc_out.state_dict())

    # 4) 전체 freeze
    for p in lora_model.parameters():
        p.requires_grad = False

    # LoRA branch (A,B)만 trainable
    for m in [lora_model.conv2_lora, lora_model.conv3_lora]:
        if m.r > 0:
            for p in m.lora_down.parameters():
                p.requires_grad = True
            for p in m.lora_up.parameters():
                p.requires_grad = True

    # FC 부분도 trainable
    for p in lora_model.fc1.parameters():
        p.requires_grad = True
    for p in lora_model.fc_out.parameters():
        p.requires_grad = True

    # Optimizer는 trainable인 것만
    lora_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, lora_model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    lora_ckpt_path = os.path.join(RESULT_DIR, "lora_cnn_best_val_loss.pth")

    history_lora = train_model(
        model=lora_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=lora_optimizer,
        criterion=criterion,
        epochs=30,
        ckpt_path=lora_ckpt_path,
        model_name="LoRA",
        device = device,
    )

    history_path = os.path.join(RESULT_DIR, "history_lora.json")
    with open(history_path, "w") as f:
        json.dump(history_lora, f, indent=2)

    lora_test_loss, lora_test_acc = eval_model(
        model=lora_model,
        test_loader=test_loader,
        criterion=criterion,
        ckpt_path=lora_ckpt_path,
        model_name="LoRA",
        device = device,
    )
    
    test_result_path = os.path.join(RESULT_DIR, "test_lora.json")
    with open(test_result_path, "w") as f:
        json.dump(
            {
                "test_loss": lora_test_loss,
                "test_acc": lora_test_acc,
            },
            f,
            indent=2,
        )
        
if __name__ == "__main__":
    main()
