from torch import nn
import torch.nn.functional as F


class ReconsClassModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim - 1, 50)   # hidden layer 1
        self.fc2 = nn.Linear(50, 50)          # hidden layer 2
        self.fc3 = nn.Linear(50, 50)          # hidden layer 3
        self.fc4 = nn.Linear(50, 1)           # output layer

    def forward(self, x):
        x = x[:, 1:]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x