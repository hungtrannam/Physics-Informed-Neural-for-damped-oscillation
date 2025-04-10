import torch
import torch.nn as nn

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class tanhsig(nn.Module):
    def forward(self, x):
        return torch.tanh(x) * torch.sigmoid(x)