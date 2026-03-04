import os
import torch

def train_model(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion,
    epochs, 
    ckpt_path, 
    model_name="Model",
    device="cpu", 
    ):
    
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        # ====== TRAIN ======
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, pred_labels = preds.max(1)
            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0.0

        # ====== VALIDATION ======
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)

                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()

                _, pred_labels = preds.max(1)
                val_correct += (pred_labels == yb).sum().item()
                val_total += yb.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{model_name} Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # best val_loss 기준으로 체크포인트 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Best {model_name} saved!")

    return history
