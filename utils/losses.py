import torch
from data.data_provider import GAMMA, OMEGA

def comp_loss(model, t, args, t_data=None, u_data=None):
    """
    Compute loss for PINN training.

    mode:
        - "PINN": physics loss + bc loss + data loss
        - "PI"  : physics loss + bc loss only
        - "NN"  : data loss only
    """
    device = t.device
    t.requires_grad = True
    u_pred = model(t)

    # Always initialize as tensor
    loss_eq = torch.tensor(0.0, dtype=torch.float32, device=device)
    loss_bc = torch.tensor(0.0, dtype=torch.float32, device=device)

    # Physics + BC loss (for PI and PINN mode)
    if args.model_type in ["PI", "PINN"]:
        u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        loss_eq = torch.mean((u_tt + 2 * GAMMA * u_t + OMEGA ** 2 * u_pred) ** 2)

        # Boundary condition at t = 0
        t_bc = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True, device=device)
        u_bc_pred = model(t_bc)
        u_bc_exact = torch.tensor([[1.0]], dtype=torch.float32, device=device)
        loss_bc_1 = torch.mean((u_bc_pred - u_bc_exact) ** 2)
        u_t_bc = torch.autograd.grad(u_bc_pred, t_bc, grad_outputs=torch.ones_like(u_bc_pred), create_graph=True)[0]
        loss_bc_2 = torch.mean((u_t_bc - 0.0) ** 2)
        loss_bc = loss_bc_1 + loss_bc_2

    # Data loss (always present for NN and PINN)
    if t_data is not None and u_data is not None:
        u_pred_data = model(t_data)
        loss_data = torch.mean((u_pred_data - u_data) ** 2)
    else:
        # synthetic exact solution
        u_exact = torch.exp(-GAMMA * t) * torch.cos(OMEGA * t)
        loss_data = torch.mean((u_pred - u_exact) ** 2)

    # Combine losses according to mode
    if args.model_type == "PI":
        total_loss = args.omega_eq * loss_eq + args.omega_bc * loss_bc
    elif args.model_type == "NN":
        total_loss = args.omega_dt * loss_data
    elif args.model_type == "PINN":
        total_loss = args.omega_eq * loss_eq + args.omega_bc * loss_bc + args.omega_dt * loss_data

    return total_loss, loss_eq, loss_bc, loss_data
