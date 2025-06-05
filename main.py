import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from model import MLP
from data_loading import load_adult_income_data
from train import train
from evaluate import evaluate
from plot import plot_losses

def main(args):
    torch.manual_seed(42)

    # Load data
    X_train, y_train, X_test, y_test, input_dim = load_adult_income_data(args.data_path)

    # Initialize model, loss, optimizer
    model = MLP(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    train_losses, test_losses = train(model, X_train, y_train, X_test, y_test, criterion, optimizer, epochs=args.epochs)

    # Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

    # Evaluate on test data
    evaluate(model, X_test, y_test)

    # Plot loss curves
    plot_losses(train_losses, test_losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on Adult Income Dataset")
    parser.add_argument('--data-path', type=str, default='dataset/adult_income_cleaned.csv', help='Path to cleaned Adult Income CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-path', type=str, default='mlp_adult_income.pth', help='Path to save the trained model')
    args = parser.parse_args()

    main(args)
