import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from src.models.scratch import BaselineCNN
from src.models.lora import LoRACNN
from src.models.adapter import AdapterCNN

from src.utils.plot_training_curves import plot_training_curves
from src.utils.plot_training_curves_smoothed import plot_training_curves_smoothed

ROOT_DIR = Path(__file__).resolve().parents[2]   # demo-repository/
RESULT_DIR = ROOT_DIR / "results"

def load_history(name: str):
    path=os.path.join(RESULT_DIR, f"history_{name}.json")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"History file not found: {name}")
    with open(path, "r") as f:
        history=json.load(f)
    return history

def load_test(name: str):
    path = RESULT_DIR / f"test_{name}.json"
    print("[DEBUG][test] looking for:", path)
    
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    with path.open("r") as f:
          test = json.load(f)
    return test

def count_trainable_params_torch(model):
    return int(np.sum([p.numel() for p in model.parameters() if p.requires_grad]))


def main():
    history_base = load_history("scratch")
    history_fft = load_history("fft")
    history_lora = load_history("lora")
    history_adapter = load_history("adapter")
    
    # Visualization for training curves
    plot_training_curves(history_base, history_fft, history_lora, history_adapter)
    plt.show()
    # Visualization for smoothed training curves
    plot_training_curves_smoothed(history_base, history_fft, history_lora,history_adapter)
    plt.show()
    
    baseline_model = BaselineCNN(num_classes=10)
    fft_model      = BaselineCNN(num_classes=10)
    lora_model     = LoRACNN(num_classes=10)  
    adapter_model  = AdapterCNN(num_classes=10)

    base_params    = count_trainable_params_torch(baseline_model)
    fft_params     = count_trainable_params_torch(fft_model)
    lora_params     = count_trainable_params_torch(lora_model)
    adapter_params = count_trainable_params_torch(adapter_model)

    # 2) 테스트 결과 로드 (train_scratch.py 등에서 json으로 저장해놨다고 가정)
    base_test = load_test("scratch")
    fft_test  = load_test("fft")
    lora_test = load_test("lora")
    ad_test   = load_test("adapter")

    base_test_acc  = base_test["test_acc"]
    base_test_loss = base_test["test_loss"]
    fft_test_acc   = fft_test["test_acc"]
    fft_test_loss  = fft_test["test_loss"]
    lora_test_acc   = lora_test["test_acc"]
    lora_test_loss  = lora_test["test_loss"]    
    adapter_test_acc  = ad_test["test_acc"]
    adapter_test_loss = ad_test["test_loss"]

    # 3) DataFrame 생성   
    df = pd.DataFrame({
        "Model": [
            "Baseline (Scratch)",
            "Full Fine-Tuning",
            "LoRA CNN",
            "Adapter CNN",
        ],
        "Trainable Params": [
            base_params,
            fft_params,
            lora_params,
            adapter_params,
        ],
        "Test Accuracy": [
            base_test_acc,
            fft_test_acc,
            lora_test_acc,
            adapter_test_acc,
        ],
        "Test Loss": [
            base_test_loss,
            fft_test_loss,
            lora_test_loss,
            adapter_test_loss,
        ]
    })

    df = df.round({"Test Accuracy": 4, "Test Loss": 4})

    print(df)  # 터미널용
    print()
    print(df.to_markdown(index=False))  # 논문/리포트용 표 형태
    
if __name__ == "__main__":
    main()