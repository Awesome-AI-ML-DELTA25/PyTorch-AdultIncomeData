import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Testing Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./graphs/loss_curve.png")
    plt.show()