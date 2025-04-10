import torch
import torch.nn as nn

class PIRNN(nn.Module):
    def __init__(self, num_hidden_layers, num_neurons, dropout_rate=0.0, rnn_type='GRU'):
        super(PIRNN, self).__init__()

        RNN_LAYER = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }

        assert rnn_type in RNN_LAYER, f"RNN type '{rnn_type}' not supported."

        # Input layer
        self.input_layer = nn.Linear(1, num_neurons)

        # RNN layers
        self.rnn = RNN_LAYER[rnn_type](
            input_size=num_neurons,
            hidden_size=num_neurons,
            num_layers=int(num_hidden_layers),
            batch_first=True,
            dropout=dropout_rate
        )

        # Output layer
        self.output_layer = nn.Linear(num_neurons, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # reshape x từ [B, input_dim=1] -> [B, seq_len=1, input_dim=1]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_layer(x)
        x = self.dropout(x)

        output, _ = self.rnn(x)
        
        # lấy output cuối cùng tại seq_len=1
        out = self.output_layer(output[:, -1, :])
        return out
