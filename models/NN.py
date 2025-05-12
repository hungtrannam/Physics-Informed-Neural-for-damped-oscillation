import torch
import torch.nn as nn
from utils.actifunc import get_activation

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class NeuralNet(nn.Module):
    """
    General Neural Network backbone supporting:
    - FeedForward (MLP)
    - RNN
    - LSTM
    - GRU
    - TransformerEncoder (added)
    """
    def __init__(self, model_type='FeedForward', num_hidden_layers=3, num_neurons=64,
                 dropout_rate=0.0, activation_name='tanh', n_heads=4, d_ff=128):
        super(NeuralNet, self).__init__()
        
        self.model_type = model_type
        self.activation = get_activation(activation_name)
        self.dropout = nn.Dropout(dropout_rate)

        if model_type == 'FeedForward':
            self.input_layer = nn.Linear(1, num_neurons)
            self.hidden_layers = nn.ModuleList([
                nn.Linear(num_neurons, num_neurons) for _ in range(int(num_hidden_layers))
            ])
            self.output_layer = nn.Linear(num_neurons, 1)

        elif model_type in ['RNN', 'LSTM', 'GRU']:
            rnn_module = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[model_type]
            self.rnn = rnn_module(input_size=1, hidden_size=num_neurons,
                                  num_layers=num_hidden_layers, batch_first=True)
            self.output_layer = nn.Linear(num_neurons, 1)

        elif model_type == 'TransformerEncoder':
            d_model = num_neurons
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True
            )
            self.embedding = nn.Linear(1, d_model)
            self.positional_encoding = PositionalEncoding(d_model)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
            self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.float()

        if self.model_type == 'FeedForward':
            x = self.activation(self.input_layer(x))
            x = self.dropout(x)
            for layer in self.hidden_layers:
                x = self.activation(layer(x))
                x = self.dropout(x)
            x = self.output_layer(x)

        elif self.model_type in ['RNN', 'LSTM', 'GRU']:
            x = x.unsqueeze(-1)  # (batch, seq_len) → (batch, seq_len, 1)
            rnn_out, _ = self.rnn(x)
            x = self.output_layer(rnn_out).squeeze(-1)

        elif self.model_type == 'TransformerEncoder':
            x = x.unsqueeze(-1)  # (batch, seq_len) → (batch, seq_len, 1)
            x = self.embedding(x)  # → (batch, seq_len, d_model)
            x = self.positional_encoding(x)
            x = self.transformer(x)  # → (batch, seq_len, d_model)
            x = self.output_layer(x).squeeze(-1)

        return x
