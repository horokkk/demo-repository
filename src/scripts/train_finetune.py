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

    os.makedirs(RESULT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loader(batch_size=64)

    baseline_ckpt = os.path.join(RESULT_DIR, "cnn_best_val_loss.pth")

    fft_model = BaselineCNN(num_classes=10).to(device)
    fft_model.load_state_dict(torch.load(baseline_ckpt, map_location=device))

    criterion = nn.CrossEntropyLoss()
    fft_optimizer = torch.optim.Adam(fft_model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    fft_ckpt = os.path.join(RESULT_DIR, "cnn_fft_best_val_loss.pth")


    history_fft = train_model(
        model=fft_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=fft_optimizer,
        criterion=criterion,
        epochs=30,
        ckpt_path=fft_ckpt,
        model_name="FFT",
        device = device,
    )
    
    history_path = os.path.join(RESULT_DIR, "history_fft.json")
    with open(history_path, "w") as f:
        json.dump(history_fft, f, indent=2)



    fft_test_loss, fft_test_acc = eval_model(
        model=fft_model,
        test_loader=test_loader,
        criterion=criterion,
        ckpt_path=fft_ckpt,
        model_name="FFT"
    )
    
    test_result_path = os.path.join(RESULT_DIR, "test_fft.json")
    with open(test_result_path, "w") as f:
        json.dump(
            {
                "test_loss": fft_test_loss,
                "test_acc": fft_test_acc,
            },
            f,
            indent=2,
        )


    
if __name__ == "__main__":
    main()
