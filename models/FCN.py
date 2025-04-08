import torch
import torch.nn as nn
from utils.actifunc import swish


class FCN(nn.Module):
    def __init__(self, num_hidden_layers, num_neurons, dropout_rate=0.0, activation_name='swish'):
        """
        Initialize the Fully Connected Neural Network (FCN) model.
        :param num_hidden_layers: Number of hidden layers in the network.
        :param num_neurons: Number of neurons in each hidden layer.
        :param dropout_rate: Dropout rate for regularization.
        :param activation_name: Activation function to be used in the hidden layers.
        """
        super(FCN, self).__init__()
        
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

        # Tạo các hidden layers
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
