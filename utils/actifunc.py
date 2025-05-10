import torch
import torch.nn as nn

class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class tanhsig(nn.Module):
    def forward(self, x):
        return torch.tanh(x) * torch.sigmoid(x)
    
def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu" or name == "swish_builtin":
        return nn.SiLU()   # Swish chuẩn đã tích hợp
    elif name == "swish":
        return swish()
    elif name == "tanhsig":
        return tanhsig()
    else:
        raise ValueError(f"Unsupported activation: {name}")
