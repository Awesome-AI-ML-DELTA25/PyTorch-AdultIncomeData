from sklearn.metrics import accuracy_score, classification_report
import torch

torch.manual_seed(42)  # For reproducibility

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        preds = (preds > 0.5).float()
        acc = accuracy_score(y_test, preds)
        print(f"✅ Final Test Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))