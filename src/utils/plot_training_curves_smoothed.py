import matplotlib.pyplot as plt
from src.utils.smooth_curves import smooth_curve


def plot_training_curves_smoothed(history_base, history_fft, history_lora, history_adapter):
    epochs_base = range(1, len(history_base["train_loss"]) + 1)
    epochs_fft  = range(1, len(history_fft["train_loss"]) + 1)
    epochs_lora  = range(1, len(history_lora["train_loss"]) + 1)
    epochs_adapter = range(1, len(history_adapter["train_loss"]) + 1)

    plt.figure(figsize=(14, 6))

    # 스타일 설정 (모델별 색상 지정)
    # Scratch: Blue, FFT: Red, Adapter: purple, LoRA: Green
    styles = [
        ("Scratch", history_base, epochs_base, "tab:blue"),
        ("FFT",     history_fft,  epochs_fft,  "tab:red"),
        ("LoRA",     history_lora,  epochs_lora,  "tab:purple"),
        ("Adapter",    history_lora, epochs_adapter, "tab:green"),
    ]

    # ===== 1. Loss 그래프 =====
    plt.subplot(1, 2, 1)
    for name, history, epochs, color in styles:
        # Train Loss (점선 --)
        # 원본 (흐리게)
        plt.plot(epochs, history["train_loss"], color=color, linestyle='--', alpha=0.2)
        # 스무딩 (진하게)
        plt.plot(epochs, smooth_curve(history["train_loss"]), color=color, linestyle='--', label=f"{name} Train")

        # Val Loss (실선 -)
        # 원본 (흐리게)
        plt.plot(epochs, history["val_loss"], color=color, linestyle='-', alpha=0.2)
        # 스무딩 (진하게)
        plt.plot(epochs, smooth_curve(history["val_loss"]), color=color, linestyle='-', linewidth=2, label=f"{name} Val")

    plt.title("Training & Validation Loss (Smoothed)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ===== 2. Accuracy 그래프 =====
    plt.subplot(1, 2, 2)
    for name, history, epochs, color in styles:
        # Train Acc (점선 --)
        plt.plot(epochs, history["train_acc"], color=color, linestyle='--', alpha=0.2)
        plt.plot(epochs, smooth_curve(history["train_acc"]), color=color, linestyle='--', label=f"{name} Train")

        # Val Acc (실선 -)
        plt.plot(epochs, history["val_acc"], color=color, linestyle='-', alpha=0.2)
        plt.plot(epochs, smooth_curve(history["val_acc"]), color=color, linestyle='-', linewidth=2, label=f"{name} Val")

    plt.title("Training & Validation Accuracy (Smoothed)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

