
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


from utils.config import get_args, set_seed
from data.data_provider import t_data, u_data
from models.PINN import PINN
from exp.exp_main import train
from utils.vis import evaluate_model
import torch
import torch.optim as optim
import datetime
import os

def main():
    run_id = f"PINN_{datetime.datetime.now():%d_%m___%H_%M}"
    output_dir = os.path.join("runs/PINN2", run_id)
    os.makedirs(output_dir, exist_ok=True)

    args = get_args(output_dir)
    set_seed(args.seed)

    model = PINN(args.num_hidden_layers, args.num_neurons, args.dropout_rate, args.activation_name)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train(model, optimizer, num_epochs=args.num_epochs, args=args, output_dir=output_dir)
    evaluate_model(model, output_dir=output_dir)

    print('Training complete!')

if __name__ == "__main__":
    main()
