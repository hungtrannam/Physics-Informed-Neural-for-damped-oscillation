import os
import torch
import numpy as np
from scipy.stats.qmc import Halton
from utils.config import set_seed

set_seed(123)

GAMMA = 0.3
OMEGA = 3.0

# Generate Halton samples in [0, 5]
num_points = 10
halton_engine = Halton(d=1, scramble=False)
t_halton_np = 5 * halton_engine.random(n=num_points)
t_data = torch.tensor(t_halton_np, dtype=torch.float32).squeeze()  # shape (N,)

# Compute u(t) and add noise
u_data = torch.exp(-GAMMA * t_data) * torch.cos(OMEGA * t_data)
noise = 0.05
u_data += noise * torch.randn_like(u_data)

# Stack and sort
data = torch.stack((t_data, u_data), dim=1)  # shape (N, 2)
data = data[data[:, 0].argsort()]           # sort by t_data

# Split again and reshape to (N, 1)
t_data = data[:, 0].unsqueeze(-1)
u_data = data[:, 1].unsqueeze(-1)

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Save to file
save_path = f'data/noise_{noise}.txt'
np.savetxt(
    save_path,
    np.hstack((t_data.numpy(), u_data.numpy())),
    header='t_data u_data',
    fmt='%.6f %.6f'
)
