# Visualization for Adapter
import matplotlib.pyplot as plt

def plot_training_curves(history_base, history_fft, history_lora, history_adapter):
    epochs_base = range(1, len(history_base["train_loss"]) + 1)
    epochs_fft  = range(1, len(history_fft["train_loss"]) + 1)
    epochs_lora  = range(1, len(history_lora["train_loss"]) + 1)
    epochs_adapter = range(1, len(history_adapter["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # ===== Loss 그래프 =====
    plt.subplot(1, 2, 1)
    plt.plot(epochs_base, history_base["train_loss"], label="Scratch Train Loss")
    plt.plot(epochs_base, history_base["val_loss"],   label="Scratch Val Loss")

    plt.plot(epochs_fft,  history_fft["train_loss"],  label="FFT Train Loss")
    plt.plot(epochs_fft,  history_fft["val_loss"],    label="FFT Val Loss")

    plt.plot(epochs_lora,  history_lora["train_loss"],  label="LoRA Train Loss")
    plt.plot(epochs_lora,  history_lora["val_loss"],    label="LoRA Val Loss")

    plt.plot(epochs_adapter, history_adapter["train_loss"], label="Adapter Train Loss")
    plt.plot(epochs_adapter, history_adapter["val_loss"],   label="Adapter Val Loss")

    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # ===== Accuracy 그래프 =====
    plt.subplot(1, 2, 2)
    plt.plot(epochs_base, history_base["train_acc"], label="Scratch Train Acc")
    plt.plot(epochs_base, history_base["val_acc"],   label="Scratch Val Acc")

    plt.plot(epochs_fft,  history_fft["train_acc"],  label="FFT Train Acc")
    plt.plot(epochs_fft,  history_fft["val_acc"],    label="FFT Val Acc")
    
    plt.plot(epochs_lora,  history_lora["train_acc"],  label="LoRA Train Acc")
    plt.plot(epochs_lora,  history_lora["val_acc"],    label="LoRA Val Acc")

    plt.plot(epochs_adapter, history_adapter["train_acc"], label="Adapter Train Acc")
    plt.plot(epochs_adapter, history_adapter["val_acc"],   label="Adapter Val Acc")

    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

