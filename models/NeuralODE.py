import torch
import torch.nn as nn
from torchdiffeq import odeint
from utils.actifunc import swish

class ODEFunc(nn.Module):
    def __init__(self, num_hidden_layers, num_neurons, dropout_rate=0.0, activation_name='tanh'):
        super(ODEFunc, self).__init__()

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
        }

        self.activation = activations[activation_name]
        self.input_layer = nn.Linear(2, num_neurons)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(num_neurons, num_neurons) for _ in range(int(num_hidden_layers))
        ])

        self.output_layer = nn.Linear(num_neurons, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, t, x):
        t_input = t.expand(x.shape[0], 1)
        xt = torch.cat([t_input, x], dim=1)
        x = self.activation(self.input_layer(xt))
        x = self.dropout(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        dxdt = self.output_layer(x)
        return dxdt

class NeuralODE(nn.Module):
    def __init__(self, num_hidden_layers, 
                 num_neurons, 
                 dropout_rate=0.0, 
                 activation_name='tanh', 
                 solver='rk4', atol=1e-6, rtol=1e-6):
        super(NeuralODE, self).__init__()
        self.ode_func = ODEFunc(num_hidden_layers, num_neurons, dropout_rate, activation_name)
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    def forward(self, x0, t):
        x = odeint(self.ode_func, x0, t, method=self.solver, atol=self.atol, rtol=self.rtol)
        return x.squeeze(-1)
