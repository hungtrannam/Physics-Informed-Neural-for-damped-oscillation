import torch
import torch.nn as nn

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)