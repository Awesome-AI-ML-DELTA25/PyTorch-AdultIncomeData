from load_data import load_dataset
from model import MLP
from torch import nn, optim
from train import train
import torch
from matplotlib import pyplot as plt
from evaluate import evaluate
import argparse
from plot import plot_losses

def main():
    parser = argparse.ArgumentParser(description="Train MLP on Adult Income Dataset (PyTorch)")
    parser.add_argument('--data-path', type=str, default='dataset/adult_cleaned.csv', help='Path to Adult Income CSV')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.007, help='Learning rate')
    args = parser.parse_args()
    X_train, y_train, X_test, y_test,X_cross,y_cross, input_dim = load_dataset(args.data_path)
    print(f"Data loaded successfully with input dimension: {input_dim}")
    model = MLP(input_dim)
    print("Model initialized successfully.")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.0006)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_losses, test_losses = train(model, X_train, y_train, X_cross, y_cross, criterion, optimizer, epochs=args.epochs, scheduler=scheduler)
    # Save the trained model
    torch.save(model.state_dict(), "mlp_adult.pth")
    print("Model saved to mlp_adult.pth")
    print()
    evaluate(model, X_test, y_test)
    evaluate(model, X_cross, y_cross)
    plot_losses(train_losses, test_losses)



if __name__ == "__main__":
    main()