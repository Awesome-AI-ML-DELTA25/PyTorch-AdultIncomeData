import argparse
from model import MLP
from data_loader import load_adult_income_data
from train import train
from evaluate import evaluate
from plot import plot_losses
import torch.nn as nn
import torch.optim as optim
import torch

torch.manual_seed(42)  # For reproducibility

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on Adult Income Dataset (PyTorch)")
    parser.add_argument('--data-path', type=str, default='data/adultIncome.csv', help='Path to Adult Income CSV')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, input_dim = load_adult_income_data(args.data_path)

    model = MLP(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses, test_losses = train(model, X_train, y_train, X_test, y_test, criterion, optimizer, epochs=args.epochs)

    # Save the trained model
    torch.save(model.state_dict(), "mlp_adultincome.pth")
    print("Model saved to mlp_adultincome.pth")

    print()
    
    plot_losses(train_losses, test_losses)
    evaluate(model, X_test, y_test)