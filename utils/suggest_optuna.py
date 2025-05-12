def suggest_args(trial, args):
    """
    Suggest hyperparameters for Optuna trial and override args.
    """

    # General common hyperparameters
    args.model_type = trial.suggest_categorical('model_type', 
        ['FeedForward', 'RNN', 'LSTM', 'GRU', 'TransformerEncoder'])
    
    args.num_hidden_layers = trial.suggest_int('num_hidden_layers', 2, 8)
    args.num_neurons = trial.suggest_categorical('num_neurons', [32, 64, 128, 256])
    args.dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.1)
    args.activation_name = trial.suggest_categorical(
        'activation_name', ['relu', 'tanh', 'gelu', 'silu', 'swish']
    )
    args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    args.weight_decay = trial.suggest_float('weight_decay', 0.0, 0.001)

    # Special params for TransformerEncoder only
    if args.model_type == 'TransformerEncoder':
        args.n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        args.d_ff = trial.suggest_categorical('d_ff', [64, 128, 256, 512])

    return args
