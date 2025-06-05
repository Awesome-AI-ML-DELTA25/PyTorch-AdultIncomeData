import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, X, y_true, threshold=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        preds = (outputs >= threshold).float()
    
    y_true_np = y_true.cpu().numpy()
    preds_np = preds.cpu().numpy()

    accuracy = accuracy_score(y_true_np, preds_np)
    precision = precision_score(y_true_np, preds_np)
    recall = recall_score(y_true_np, preds_np)
    f1 = f1_score(y_true_np, preds_np)

    print(f"Evaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


