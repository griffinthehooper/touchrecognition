from torch import nn
import torch
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 5, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(5, 10, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(10 * 21 * 5, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x