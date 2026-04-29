import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(196, 32),
            nn.LayerNorm(normalized_shape=[32]),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.LayerNorm(normalized_shape=[16]),
            nn.LeakyReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(16, 10),
        )

    def forward(self, x):
        x = F.avg_pool2d(x, 2, 2)
        x = torch.flatten(x, start_dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.head(x)
        return x
    
if __name__ == "__main__":
    model = Model()
    print(model)
    num_param = 0
    for param in model.parameters():
        num_param += param.numel()
    print("num_param:", num_param)