import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_attributes=3):  # smiling, eyeglasses, male
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # input size = 64×16×16
        self.fc2 = nn.Linear(128, num_attributes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

