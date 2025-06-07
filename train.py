from tqdm import tqdm
import torch

torch.manual_seed(42)  # For reproducibility

def train(model, X_train, y_train, X_test, y_test, criterion, optimizer, epochs=100, scheduler=None):
    model.train()
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        # --- Training step ---
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # --- Evaluation step ---
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss = criterion(test_preds, y_test).item()
            test_losses.append(test_loss)
        
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

        # Step the scheduler with validation loss
        if scheduler is not None:
            scheduler.step(test_loss)
        model.train()

    return train_losses, test_losses
