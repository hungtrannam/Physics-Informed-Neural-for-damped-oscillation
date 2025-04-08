import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import datetime
from torch.utils.data import DataLoader, TensorDataset

from utils.config import get_args, GAMMA, OMEGA, t_data, u_data
from utils.utils import evaluate_model_NeuralODE, evaluate_loss_FCN, set_seed
from models.NeuralODE import NeuralODE




def physics_loss(model, t, omega_dt, t_data=None, u_data=None):
    x0 = torch.tensor([[1.0]])  # Điều kiện đầu
    u_pred = model(x0, t)

    if t_data is not None and u_data is not None:
        u_pred_data = model(x0, t_data)
        loss_data = torch.mean((u_pred_data - u_data) ** 2)
    else:
        u_exact = torch.exp(-GAMMA * t) * torch.cos(OMEGA * t)
        loss_data = torch.mean((u_pred - u_exact)**2)

    total_loss = omega_dt * loss_data
    return total_loss, loss_data

def train_NeuralODE(model, optimizer, num_epochs=10000, omega_dt=1, output_dir='runs', use_Vis=False):
    t_train = torch.linspace(0.0, 15.0, 500)
    dataset = TensorDataset(t_train)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    frames = []
    loss_history, data_loss_history = [], []

    print("Training NeuralODE with Adam...")
    for epoch in range(num_epochs):
        for batch in dataloader:
            t_batch = batch[0]
            optimizer.zero_grad()
            loss, loss_data = physics_loss(model, t_batch, omega_dt, t_data, u_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.1)
            optimizer.step()
        scheduler.step(loss.item())

        loss_history.append(loss.item())
        data_loss_history.append(loss_data.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}, LR = {scheduler.get_last_lr()[0]}')
            if use_Vis:
                img_dir = os.path.join(output_dir, "frames")
                os.makedirs(img_dir, exist_ok=True)
                t_test = torch.linspace(0, 15, 200)
                x0 = torch.tensor([[1.0]])
                u_pred = model(x0, t_test).detach().numpy()
                u_exact = np.exp(-GAMMA * t_test.numpy()) * np.cos(OMEGA * t_test.numpy())
                plt.figure(figsize=(10, 6))
                plt.scatter(t_test.numpy(), u_pred, label=f'NeuralODE Epoch {epoch}', s=10)
                plt.plot(t_test.numpy(), u_exact, label='Exact Solution', color='red')
                plt.legend()
                plt.xlabel('t')
                plt.ylabel('u(t)')
                plt.ylim(-1, 1)
                plt.title(f'NeuralODE Damped Oscillation at Epoch {epoch}')
                
                frame_path = os.path.join(img_dir, f'frame_{epoch}.png')
                plt.savefig(frame_path)
                frames.append(imageio.v2.imread(frame_path))
                plt.close()

                gif_path = os.path.join(output_dir, 'solution.gif')
                imageio.mimsave(gif_path, frames, duration=5)
                print(f'GIF saved to {gif_path}')
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    evaluate_loss_FCN(num_epochs, loss_history, data_loss_history, output_dir)

def main():
    run_id = f"NeuralODE_{datetime.datetime.now().strftime('%d_%m___%H_%M')}"
    output_dir = os.path.join("runs/NeuralODE", run_id)
    os.makedirs(output_dir, exist_ok=True)

    args = get_args(output_dir)
    set_seed(args.seed)

    ################################################################################
    # Initialize the NeuralODE model
    model = NeuralODE(num_hidden_layers=args.num_hidden_layers,
                      num_neurons=args.num_neurons,
                      dropout_rate=args.dropout_rate,
                      activation_name=args.activation_name,)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print('Training NeuralODE model...')
    train_NeuralODE(model, optimizer, 
                    num_epochs=args.num_epochs, 
                    omega_dt=args.omega_dt,
                    output_dir=output_dir)
    
    evaluate_model_NeuralODE(model, output_dir=output_dir)
    print('Training complete!')

if __name__ == "__main__":
    main()
