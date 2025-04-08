import argparse
import json
import torch
import torch
import numpy as np


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.random.manual_seed(seed)
    


def get_args(output_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--activation_name', type=str, default='swish')
    parser.add_argument('--num_hidden_layers', type=int, default=6)
    parser.add_argument('--dropout_rate', type=float, default=0.000)
    parser.add_argument('--num_neurons', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--omega_eq', type=float, default=1)
    parser.add_argument('--omega_bc', type=float, default=1)
    parser.add_argument('--omega_dt', type=float, default=1)

    args = parser.parse_args()

    # Convert args to dictionary and save to JSON
    args_dict = vars(args)
    with open(output_dir+'/config.json', 'w') as f:
        json.dump(args_dict, f, indent=4)

    return args
