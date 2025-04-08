import torch.nn as nn
from utils.actifunc import swish
import torch

class PINN(nn.Module):
    def __init__(self, num_hidden_layers, num_neurons, dropout_rate=0.0, activation_name='tanh'):
        super(PINN, self).__init__()
        
        activations = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'sine': torch.sin,
            
            'swish': swish(),
            'logsigmoid': nn.LogSigmoid(),
            'softplus': nn.Softplus(),
            'elist': nn.ELU(),
            'sel': nn.SELU(),
            'elu': nn.ELU(),

            # 'softmax': nn.Softmax(dim=1),
            # 'leaky_relu': nn.LeakyReLU(),
            # 'gelu': nn.GELU(),
            # 'elu': nn.ELU(),
            # 'prelu': nn.PReLU(),

        }
        
        self.activation = activations[activation_name]

        self.input_layer = nn.Linear(1, num_neurons)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(num_neurons, num_neurons) for _ in range(int(num_hidden_layers))
        ])

        self.output_layer = nn.Linear(num_neurons, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x
