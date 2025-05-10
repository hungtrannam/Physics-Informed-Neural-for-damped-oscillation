import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from torch.utils.data import DataLoader, TensorDataset
from utils.vis import evaluate_loss_PINN
from data.data_provider import GAMMA, OMEGA, t_data, u_data

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.losses import comp_loss

def train(model, optimizer, num_epochs=10000, args=None, output_dir='runs', use_Vis=False):
    t_train = torch.linspace(0.0, 15.0, 500).reshape(-1, 1)
    dataset = TensorDataset(t_train)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    frames = []
    loss_history, data_loss_history = [], []

    print("Training with Adam...")
    for epoch in range(num_epochs):
        for batch in dataloader:
            t_batch = batch[0]
            optimizer.zero_grad()
            total_loss, loss_eq, loss_bc, loss_data = comp_loss(
                model, t_batch, t_data=t_data, u_data=u_data, args=args)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
        scheduler.step(total_loss.item())

        loss_history.append(total_loss.item())
        data_loss_history.append(loss_data.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {total_loss.item():.4e}, LR = {scheduler.get_last_lr()[0]:.2e}')
            if use_Vis:
                img_dir = os.path.join(output_dir, "frames")
                os.makedirs(img_dir, exist_ok=True)
                t_test = torch.linspace(0, 15, 200).reshape(-1, 1)
                u_pred = model(t_test).detach().numpy()
                u_exact = np.exp(-GAMMA * t_test.numpy()) * np.cos(OMEGA * t_test.numpy())
                plt.figure(figsize=(10, 6))
                plt.scatter(t_test.numpy(), u_pred, label=f'Epoch {epoch}', s=10)
                plt.plot(t_test.numpy(), u_exact, label='Exact Solution', color='red')
                plt.legend()
                plt.xlabel('t')
                plt.ylabel('u(t)')
                plt.ylim(-1, 1)
                plt.title(f'Damped Oscillation at Epoch {epoch}')
                frame_path = os.path.join(img_dir, f'frame_{epoch}.png')
                plt.savefig(frame_path)
                frames.append(imageio.v2.imread(frame_path))
                plt.close()

                gif_path = os.path.join(output_dir, 'training_evolution.gif')
                imageio.mimsave(gif_path, frames, duration=5)
                print(f'GIF saved to {gif_path}')

    print("Fine-tuning with LBFGS...")
    optimizer_lbfgs = optim.LBFGS(model.parameters(),
                                  lr=0.001,
                                  max_iter=500,
                                  max_eval=5000,
                                  tolerance_grad=1e-9,
                                  tolerance_change=1e-9,
                                  history_size=50,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss, _, _, _ = comp_loss(
            model, t_train, t_data=t_data, u_data=u_data, mode="PINN"
        )
        total_loss.backward()
        return total_loss

    for _ in range(500):
        optimizer_lbfgs.step(closure)

    final_loss, _, _, _ = comp_loss(
        model, t_train, t_data=t_data, u_data=u_data, mode="PINN"
    )
    print(f'Final loss after LBFGS: {final_loss.item():.4e}')

    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    evaluate_loss_PINN(num_epochs, loss_history, data_loss_history, output_dir)

def test(model, output_dir, plot=True):
    model.eval()
    t_test = torch.linspace(0.0, 15.0, 500).reshape(-1, 1)
    u_pred = model(t_test).detach().numpy()
    t_np = t_test.numpy()
    u_exact = np.exp(-GAMMA * t_np) * np.cos(OMEGA * t_np)

    mse = np.mean((u_pred - u_exact)**2)
    l2_relative = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    print(f'Test MSE: {mse:.6e}, L2 Relative Error: {l2_relative:.6e}')

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(t_np, u_exact, label='Exact Solution', color='red')
        plt.scatter(t_np, u_pred, label='Model Prediction', s=10)
        plt.xlabel('t')
        plt.ylabel('u(t)')
        plt.legend()
        plt.title('Model Test Prediction vs Exact')
        plt.savefig(os.path.join(output_dir, 'test_result.png'))
        plt.close()

    return mse, l2_relative
