# 🧠 Physics-Informed Neural Network for Damped Oscillation

This project implements a **Physics-Informed Neural Network (PINN)** to solve the damped harmonic oscillator differential equation using PyTorch. The model incorporates physics laws into its training by embedding them into the loss function.

---

## 📘 Problem Statement

We aim to solve the second-order ODE:

\[
\frac{d^2u}{dt^2} + 2\gamma \frac{du}{dt} + \omega^2 u = 0
\]

With initial conditions:

\[
u(0) = 1,\quad u'(0) = 0
\]

The exact solution is:

\[
u(t) = e^{-\gamma t} \cos(\omega t)
\]


---

## ⚙️ Loss Components

Total loss:

\[
\mathcal{L}_{\text{total}} = \omega_{\text{eq}} \cdot \mathcal{L}_{\text{eq}} + \omega_{\text{bc}} \cdot \mathcal{L}_{\text{bc}} + \omega_{\text{dt}} \cdot \mathcal{L}_{\text{data}}
\]

Where:
- `loss_eq`: Equation loss from residuals of the ODE
- `loss_bc`: Initial condition mismatch
- `loss_data`: (Optional) Supervised loss from real data

---

## 🚀 Usage

### 1. Clone and install dependencies

```bash
git clone https://github.com/hungtrannam/Physics-Informed-Neural-for-damped-oscillation.git
cd Physics-Informed-Neural-for-damped-oscillation
pip install -r requirements.txt

### 2. Run the main script

```python main_PINN2.py```

