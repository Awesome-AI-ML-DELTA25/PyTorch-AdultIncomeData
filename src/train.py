from tqdm import tqdm
import torch

torch.manual_seed(7)

def train(model, X_train, y_train, X_test, y_test, criterion, optimizer, epochs, scheduler=None):

    # Incase of a scheduler, we have to vary the lr
    
    model.train()
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        optimizer.zero_grad()   # Set gradients to zero
        outputs = model(X_train)    # Get the outputs on training data
        loss = criterion(outputs, y_train)  # Compare loss (e.g MSE etc)
        loss.backward() # Backward propogation to update values
        optimizer.step()    # Update weight
        train_losses.append(loss.item())    # Add loss to list

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)  # Find the values in testing data
            test_loss = criterion(test_preds, y_test).item()    # Find loss
            test_losses.append(test_loss)   # Append loss

        # If scheduler is present, then 
        if scheduler is not None:
            scheduler.step(test_loss)
            
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

    return train_losses, test_losses
