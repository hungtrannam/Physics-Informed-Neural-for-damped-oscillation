# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from utils.data_provider import GAMMA, OMEGA, t_data, u_data


def evaluate_model(model, output_dir):
    t_test = torch.linspace(0, 15, 200).reshape(-1, 1)
    u_pred = model(t_test).detach().numpy()
    u_exact = np.exp(-GAMMA * t_test.numpy()) * np.cos(OMEGA * t_test.numpy())

    mse = mean_squared_error(u_exact, u_pred)
    print(f'Mean Squared Error: {mse:.6f}')

    plt.figure(figsize=(10, 6))
    plt.scatter(t_test.numpy(), u_pred, label='PINN Prediction', color='blue', s=10)
    plt.scatter(t_data, u_data, label='Collocation Data', color='red', s=15)
    plt.plot(t_test.numpy(), u_exact, label='Exact Solution', color='red', linewidth=2, alpha=0.5)
    plt.xlabel('Time (t)')
    plt.ylabel('Displacement (u)')
    plt.title(f'Damped Oscillation using PINN (MSE: {mse:.6f})')
    plt.legend()
    plt.ylim(-1, 1)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'results.pdf'))
    plt.close()


def evaluate_model_NeuralODE(model, output_dir):
    t_test = torch.linspace(0, 15, 200)
    x0 = torch.tensor([[1.0]])
    
    with torch.no_grad():
        u_pred = model(x0, t_test).detach().numpy()
    
    u_exact = np.exp(-GAMMA * t_test.numpy()) * np.cos(OMEGA * t_test.numpy())
    mse = mean_squared_error(u_exact, u_pred)

    print(f'Mean Squared Error: {mse:.6f}')

    plt.figure(figsize=(10, 6))
    plt.scatter(t_test.numpy(), u_pred, label='NeuralODE Prediction', color='blue', s=10)
    plt.scatter(t_data, u_data, label='Collocation Data', color='red', s=15)
    plt.plot(t_test.numpy(), u_exact, label='Exact Solution', color='red', linewidth=2, alpha=0.5)
    plt.xlabel('Time (t)')
    plt.ylabel('Displacement (u)')
    plt.title(f'Damped Oscillation using NeuralODE (MSE: {mse:.6f})')
    plt.legend()
    plt.ylim(-1, 1)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'results.pdf'))
    plt.close()



def evaluate_loss_PINN(num_epochs, loss_history, data_loss_history, bc_loss_history, eq_loss_history, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), data_loss_history, label='Data Loss', color='red', alpha=0.5)
    plt.plot(range(num_epochs), eq_loss_history, label='Physics Loss', color='blue', alpha=0.5)
    plt.plot(range(num_epochs), bc_loss_history, label='Boundary Loss', color='green', alpha=0.5)
    plt.plot(range(num_epochs), loss_history, label='Total Loss', color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Components during Training (PINN)')
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'loss.pdf'))
    plt.close()


def evaluate_loss_FCN(num_epochs, loss_history, data_loss_history, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), data_loss_history, label='Data Loss', color='red', alpha=0.5)
    plt.plot(range(num_epochs), loss_history, label='Total Loss', color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Components during Training (FCN)')
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'loss.pdf'))
    plt.close()
