import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses, name):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Testing Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./graphs/loss_curve_{name}.png")

def plot_combined_losses(all_train_losses, all_test_losses):
    plt.figure(figsize=(10, 6))
    
    for model_name in all_train_losses:
        plt.plot(all_train_losses[model_name], label=f"{model_name} - Train")
        plt.plot(all_test_losses[model_name], label=f"{model_name} - Test")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./graphs/comparison_loss.png")

def plot_accuracies(accuracies_dict):
    model_names = list(accuracies_dict.keys())
    accuracies = list(accuracies_dict.values())

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen'])
    plt.ylim(0.85, 0.86)  # 👈 Set Y-axis limits here
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison (Zoomed In)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.0002, f"{acc:.4f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("./graphs/accuracy_comparison.png")

