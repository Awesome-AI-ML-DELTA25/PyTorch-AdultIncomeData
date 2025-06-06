import torch.nn as nn
import torch.nn.functional as torch_fn
import torch

torch.manual_seed(7)


# Custom Mish activation
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch_fn.softplus(x))

# Perform model agnostic training, using nn.Module class
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__() # Initialise parent constructor class

        self.mish = Mish()

        # Various Model layers
        self.input = nn.Linear(input_dim, 100)
        self.hidden_1 = nn.Linear(100, 50)
        self.hidden_2 = nn.Linear(50, 25)
        self.output = nn.Linear(25, 1)

        # To be able to use PReLU later
        self.prelu = nn.PReLU()

    
    def forward(self, X):

        X = self.input(X)

        X = torch_fn.relu(X)

        X = self.hidden_1(X)

        X = torch_fn.relu(X)

        X = self.hidden_2(X)

        #X = self.hidden_2(X)

        #X = self.prelu(X)

        X = torch_fn.relu(X)

        X = self.output(X)

        X = torch_fn.sigmoid(X)

        return X
    
    '''
    def forward(self, X):
        X = self.input(X)
        X = self.mish(X)

        X = self.hidden_1(X)
        X = self.mish(X)

        X = self.hidden_2(X)
        X = self.mish(X)

        X = self.hidden_3(X)
        X = self.mish(X)

        X = self.hidden_4(X)
        X = self.mish(X)

        X = self.output(X)
        X = torch.sigmoid(X)  # For binary classification
    
        return X
    '''