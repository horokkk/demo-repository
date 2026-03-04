import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)   # (B,32,20,87)
        x = self.conv2(x)   # (B,64,10,43)
        x = self.conv3(x)   # (B,128,5,21)
        x = self.gap(x)     # (B,128,1,1)
        x = x.view(x.size(0), -1)  # (B,128)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

