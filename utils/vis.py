# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from data.data_provider import GAMMA, OMEGA, t_data, u_data

def set_plot_style():
    plt.rcParams.update({
        'font.size': 15,
        'font.family': 'serif',
        'mathtext.fontset': 'cm', 
        'axes.linewidth': 1,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'legend.frameon': False,
        'legend.fontsize': 13,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
    })


def evaluate_loss(args, loss_history=None, data_loss_history=None, bc_loss_history=None, eq_loss_history=None):
    set_plot_style()
    plt.figure(figsize=(10, 6))
    plt.plot(range(args.num_epochs), data_loss_history, label='Data Loss', color='red', alpha=0.5)
    if eq_loss_history is not None:
        plt.plot(range(args.num_epochs), eq_loss_history, label='Physics Loss', color='blue', alpha=0.5)
    if bc_loss_history is not None:
        plt.plot(range(args.num_epochs), bc_loss_history, label='Boundary Loss', color='green', alpha=0.5)
    plt.plot(range(args.num_epochs), loss_history, label='Total Loss', color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Components during Training of {args.model_type}')
    plt.legend()
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, 'loss.pdf'))
    plt.close()