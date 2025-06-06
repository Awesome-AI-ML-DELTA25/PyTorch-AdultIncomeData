import argparse
from model import MLP
from load_data import load_adultCensus_data
from train import train
from evaluate import evaluate
from plot import plot_losses
import torch.nn as nn
import torch.optim as optim
import torch

torch.manual_seed(7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on Adult Census Data")
    parser.add_argument('--data-path', type=str, default='data/Cleaned_Adult_Census.csv', help='Path to Adult Census Data')
    parser.add_argument('--epochs', type=int, default=170, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, input_dim = load_adultCensus_data(args.data_path)

    model = MLP(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Having a variable learning rate can improve performance
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses, test_losses = train(model, X_train, y_train, X_test, y_test, criterion, optimizer, epochs=args.epochs)

    # Save the trained model
    torch.save(model.state_dict(), "mlp_AdultCensus.pth")
    print("Model saved to mlp_AdultCensus.pth")

    print()
    
    plot_losses(train_losses, test_losses)
    evaluate(model, X_test, y_test)
