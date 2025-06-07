import torch.nn as nn
import torch.nn.functional as F
import torch


# Perform model agnostic training, using nn.Module class
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__() # Initialise parent constructor class

        # Various Model layers
        self.input_layer = nn.Linear(input_dim, 250)
        self.hidden1 = nn.Linear(250, 180)
        self.hidden2 = nn.Linear(180, 100)
        self.hidden3 = nn.Linear(100, 25)
        self.output_layer = nn.Linear(25, 1)


    def forward(self, x):
        # Pass input through the first linear layer
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))

        # Pass through the output layer
        x = self.output_layer(x)
        # Apply sigmoid activation to get output between 0 and 1
        return x
