# Summary for Adapter
import pandas as pd
import numpy as np

def count_trainable_params_torch(model):
    return int(np.sum([p.numel() for p in model.parameters() if p.requires_grad]))

# 1) Baseline 결과
base_params = count_trainable_params_torch(baseline_model)

# 2) FFT 결과
fft_params = count_trainable_params_torch(fft_model)

# 3) Adapter 결과
adapter_params = count_trainable_params_torch(adapter_model)

# DataFrame 생성
df = pd.DataFrame({
    "Model": ["Baseline (Scratch)", "Full Fine-Tuning", "Adapter CNN"],
    "Trainable Params": [base_params, fft_params, adapter_params],
    "Test Accuracy": [base_test_acc, fft_test_acc, adapter_test_acc],
    "Test Loss": [base_test_loss, fft_test_loss, adapter_test_loss]
})

df = df.round({"Test Accuracy": 4, "Test Loss": 4})
display(df)

print(df.to_markdown(index=False))
