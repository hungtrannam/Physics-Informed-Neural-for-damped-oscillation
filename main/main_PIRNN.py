import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from data.data_provider import GAMMA, OMEGA, t_data, u_data
from utils.config import get_args, set_seed
from utils.vis import evaluate_model, evaluate_loss_PINN
from models.PIRNN import PIRNN


import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from torch.utils.data import DataLoader, TensorDataset
import datetime



def physics_loss(model, t, omega_eq, omega_bc, omega_dt, t_data=None, u_data=None):
    t.requires_grad = True
    u_pred = model(t)

    u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]

    loss_eq = torch.mean((u_tt + 2*GAMMA*u_t + OMEGA**2*u_pred)**2)

    if t_data is not None and u_data is not None:
        u_pred_data = model(t_data)
        loss_data = torch.mean((u_pred_data - u_data) ** 2)
    else:
        u_exact = torch.exp(-GAMMA * t) * torch.cos(OMEGA * t)
        loss_data = torch.mean((u_pred - u_exact)**2)

    t_bc = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    u_bc_pred = model(t_bc)
    u_bc_exact = torch.tensor([[1.0]], dtype=torch.float32)
    loss_bc_1 = torch.mean((u_bc_pred - u_bc_exact)**2)
    u_t_bc = torch.autograd.grad(u_bc_pred, t_bc, grad_outputs=torch.ones_like(u_bc_pred), create_graph=True)[0]
    loss_bc_2 = torch.mean((u_t_bc - 0.0)**2)
    loss_bc = loss_bc_1 + loss_bc_2

    total_loss = omega_eq * loss_eq + omega_bc * loss_bc + omega_dt * loss_data
    return total_loss, loss_eq, loss_bc, loss_data

def train_PINN(model, optimizer, num_epochs=10000, lr_scheduler=None, omega_eq=1, omega_bc=1, omega_dt=1, output_dir='runs', use_Vis=False):
    t_train = torch.linspace(0.0, 15.0, 500).reshape(-1, 1)
    dataset = TensorDataset(t_train)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    if lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=.5, patience=10, verbose=True
        )
    elif lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=.5)
    else:
        scheduler = None


    frames = []
    loss_history, data_loss_history, bc_loss_history, eq_loss_history = [], [], [], []

    print("Training with Adam...")
    for epoch in range(num_epochs):
        for batch in dataloader:
            t_batch = batch[0]
            optimizer.zero_grad()
            loss_pinn, loss_eq, loss_bc, loss_data = physics_loss(model, t_batch, omega_eq, omega_bc, omega_dt, t_data, u_data)
            loss_pinn.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.1)
            optimizer.step()
        if lr_scheduler == 'plateau':
            scheduler.step(loss_pinn.item())
        elif lr_scheduler == 'step':
            scheduler.step()


        loss_history.append(loss_pinn.item())
        data_loss_history.append(loss_data.item())
        bc_loss_history.append(loss_bc.item())
        eq_loss_history.append(loss_eq.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss_pinn.item()}, LR = {scheduler.get_last_lr()[0]}')
            if use_Vis:
                img_dir = os.path.join(output_dir, "frames")
                os.makedirs(img_dir, exist_ok=True)
                t_test = torch.linspace(0, 15, 200).reshape(-1, 1)
                u_pred = model(t_test).detach().numpy()
                u_exact = np.exp(-GAMMA * t_test.numpy()) * np.cos(OMEGA * t_test.numpy())
                plt.figure(figsize=(10, 6))
                plt.scatter(t_test.numpy(), u_pred, label=f'{output_dir} Epoch {epoch}', s=10)
                plt.plot(t_test.numpy(), u_exact, label='Exact Solution', color='red')
                plt.legend()
                plt.xlabel('t')
                plt.ylabel('u(t)')
                plt.ylim(-1, 1)
                plt.title(f'{output_dir} Damped Oscillation Solution at Epoch {epoch}')
                
                frame_path = os.path.join(img_dir, f'frame_{epoch}.png')
                plt.savefig(frame_path)
                frames.append(imageio.v2.imread(frame_path))
                plt.close()

                gif_path = os.path.join(output_dir, f'{output_dir}_solution.gif')
                imageio.mimsave(gif_path, frames, duration=5)
                print(f'GIF saved to {gif_path}')

    def closure():
        optimizer_lbfgs.zero_grad()
        loss, _, _, _ = physics_loss(model, t_train, omega_eq, omega_bc, omega_dt, t_data, u_data)
        loss.backward()
        return loss

    print("Fine-tuning with LBFGS...")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), 
                                  lr=.001, 
                                  max_iter=500, 
                                  max_eval=5000, 
                                  tolerance_grad=1e-9, 
                                  tolerance_change=1e-9, 
                                  history_size=50, 
                                  line_search_fn="strong_wolfe")
    for _ in range(5000):
        optimizer_lbfgs.step(closure)

    final_loss, _, _, _ = physics_loss(model, t_train, omega_eq, omega_bc, omega_dt, t_data, u_data)
    print(f'Final loss after LBFGS: {final_loss.item()}')

    
    # Save the model
    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    evaluate_loss_PINN(num_epochs, loss_history, data_loss_history, bc_loss_history, eq_loss_history, output_dir)

def main():
    run_id = f"PIRNN_{datetime.datetime.now().strftime('%d_%m___%H_%M')}"
    output_dir = os.path.join("runs/PIRNN", run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Get the arguments
    args = get_args(output_dir)
    

    # Set seed for reproducibility
    set_seed(args.seed)
    # Create the model and optimizer
    model = PIRNN(num_hidden_layers=args.num_hidden_layers,
                  num_neurons=args.num_neurons,
                  dropout_rate=args.dropout_rate,
                  rnn_type=args.rnn_type)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    

    print('Training model...')
    train_PINN(model, optimizer, 
               num_epochs=args.num_epochs, 
                lr_scheduler=args.lr_scheduler,
               # batch_size=args.batch_size, 
               omega_eq=args.omega_eq, 
               omega_bc = args.omega_bc, 
               omega_dt=args.omega_dt,
               output_dir=output_dir)
    
    # Evaluate the model
    evaluate_model(model,output_dir=output_dir)
    print('Training complete!')


if __name__ == "__main__":
    main()
