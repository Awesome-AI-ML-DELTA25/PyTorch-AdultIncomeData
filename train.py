import torch
from tqdm import tqdm

def train(model, X_train, y_train, X_test, y_test, criterion, optimizer, epochs=100):
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

        tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    return train_losses, test_losses
