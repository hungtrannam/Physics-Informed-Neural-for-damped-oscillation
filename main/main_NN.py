
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


from utils.config import get_args, set_seed
from models.NN import NeuralNet
from exp.exp_main import train, test
import torch.optim as optim
import datetime
import os
import json

import optuna
from utils.suggest_optuna import suggest_args

def objective(trial):
    args = get_args()
    args = suggest_args(trial, args)

    run_id = f"{args.model_type}_{datetime.datetime.now():%d_%m___%H_%M}"
    args.output_dir = os.path.join(f"runs_optuna/{args.model_type}", run_id)
    os.makedirs(args.output_dir, exist_ok=True)

    # Lưu cấu hình
    with open(args.output_dir + '/config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    set_seed(args.seed)

    model = NeuralNet(args.num_hidden_layers,
                      args.num_neurons,
                      args.dropout_rate,
                      args.activation_name)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    train(model, optimizer, args=args, plot=False)
    val_loss = test(model, args, plot=False)

    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters:")
    print(study.best_params)

  # ========================
    # Train lại với best_params
    # ========================
    print("\nTraining final model with best hyperparameters...")

    args = get_args()
    # Ghi đè các tham số bằng best_params từ Optuna
    for key, value in study.best_params.items():
        setattr(args, key, value)

    run_id = f"best_{args.model_type}_{datetime.datetime.now():%d_%m___%H_%M}"
    args.output_dir = os.path.join(f"runs_optuna/{args.model_type}", run_id)
    os.makedirs(args.output_dir, exist_ok=True)

    # Lưu config
    with open(args.output_dir + '/config_best.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    set_seed(args.seed)

    model = NeuralNet(
        model_type=args.model_type,
        num_hidden_layers=args.num_hidden_layers,
        num_neurons=args.num_neurons,
        dropout_rate=args.dropout_rate,
        activation_name=args.activation_name,
        n_heads=getattr(args, "n_heads", 4),  # fallback nếu không phải Transformer
        d_ff=getattr(args, "d_ff", 128)
    )

    
    optimizer = optim.Adam(model.parameters(),
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay)

    train(model, optimizer, args=args, plot=True)
    test(model, args, plot=True)

    print(f"Final trained model with best params saved to {args.output_dir}")
  
