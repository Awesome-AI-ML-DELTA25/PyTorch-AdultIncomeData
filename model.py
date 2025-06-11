import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(42)  # For reproducibility

# Perform model agnostic training, using nn.Module class
class OldMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__() # Initialise parent constructor class

        # Various Model layers
        self.input_layer = nn.Linear(input_dim, 100)
        self.hidden_layer_1 = nn.Linear(100, 50)
        self.hidden_layer_2 = nn.Linear(50, 25)
        self.output_layer = nn.Linear(25, 1)


    def forward(self, x):
        # Pass input through the first linear layer
        x = self.input_layer(x)
        # Apply ReLU activation
        x = F.relu(x)
        # Pass through the first hidden layer
        x = self.hidden_layer_1(x)
        # Apply ReLU activation
        x = F.relu(x)
        # Pass through the second hidden layer
        x = self.hidden_layer_2(x)
        # Apply ReLU activation
        x = F.relu(x)
        # Pass through the output layer
        x = self.output_layer(x)
        # Apply sigmoid activation to get output between 0 and 1
        x = F.sigmoid(x)
        return x

class NewMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__() # Initialise parent constructor class

        # Various Model layers
        self.input_layer = nn.Linear(input_dim, 128)
        self.hidden_layer_1 = nn.Linear(128, 64)
        self.hidden_layer_2 = nn.Linear(64, 32)
        self.hidden_layer_3 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)


    def forward(self, x):
        # Pass input through the first linear layer
        x = self.input_layer(x)
        # Apply ReLu activation
        x = F.relu(x)
        # Pass through the first hidden layer
        x = self.hidden_layer_1(x)
        # Apply ReLU activation
        x = F.relu(x)
        # Pass through the second hidden layer
        x = self.hidden_layer_2(x)
        # Apply ReLU activation
        x = F.relu(x)
        # Pass through the third hidden layer
        x = self.hidden_layer_3(x)
        # Apply ReLU activation
        x = F.relu(x)
        # Pass through the output layer
        x = self.output_layer(x)
        # Apply sigmoid activation to get output between 0 and 1
        x = F.sigmoid(x)
        return x