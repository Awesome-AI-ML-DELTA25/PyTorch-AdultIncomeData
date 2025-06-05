import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)   # First hidden layer with 100 neurons
        self.fc2 = nn.Linear(100, 50)          # Second hidden layer with 50 neurons
        self.fc3 = nn.Linear(50, 25)           # Third hidden layer with 25 neurons
        self.output = nn.Linear(25, 1)         # Output layer with 1 neuron

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.output(x))  # Sigmoid for binary classification probability output
        return x
