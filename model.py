import torch
from torch import nn


class M_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 验证网络正确性
if __name__ == '__main__':
    model = M_CIFAR10()
    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    print(output.shape)