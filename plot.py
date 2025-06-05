import os
import matplotlib.pyplot as plt
plt.use("Agg")  # Use non-GUI backend

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Testing Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs("./graphs", exist_ok=True)  # create folder if not exists
    plt.savefig("./graphs/loss_curve.png")
    plt.show()
    print("Loss curve saved to ./graphs/loss_curve.png")