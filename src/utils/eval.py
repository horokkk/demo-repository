import torch 

def eval_model(
    model, 
    test_loader, 
    criterion, 
    ckpt_path, 
    model_name="Model",
    device="cpu", 
    ):
    # best weight 로드
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)
            test_loss += loss.item()

            _, pred_labels = preds.max(1)
            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)

    test_loss = test_loss / len(test_loader)
    test_acc = correct / total if total > 0 else 0.0

    print(f"== {model_name} Test ==")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    return test_loss, test_acc
