# Physics-Informed Neural Network for Damped Oscillation

This project implements a **Physics-Informed Neural Network (PINN)** using PyTorch to solve the damped harmonic oscillator by embedding physical laws directly into the learning process. It combines the flexibility of neural networks with the constraints of differential equations to model physical phenomena accurately.

---

## Problem Statement

We aim to solve the second-order ordinary differential equation:

```text
    d²u/dt² + 2γ du/dt + ω² u = 0
```

With initial conditions:

```text
    u(0) = 1
    u'(0) = 0
```

The exact solution is:

```text
    u(t) = exp(-γt) · cos(ωt)
```

This type of equation models many real-world systems such as:
- Mass-spring-damper mechanical systems
- Electrical RLC circuits
- Vibrating structures with damping

---

## Loss Function

The total loss function used in training is a combination of physics-informed constraints and data (if available):

```text
    L_total = ω_eq · L_eq + ω_bc · L_bc + ω_dt · L_data
```

- **L_eq**: Residual of the differential equation.
- **L_bc**: Loss enforcing initial conditions.
- **L_data** *(optional)*: Supervised data loss (if available).

---

## Model Architecture

The PINN model is a fully-connected feedforward neural network (MLP) consisting of:

- **Input layer:** maps scalar time input `t` to hidden dimension
- **Hidden layers:** configurable number and width, using chosen activation
- **Output layer:** outputs predicted scalar value `u(t)`
- **Dropout:** applied after each layer (configurable)

Supported activation functions include:
- `tanh`, `sigmoid`, `logsigmoid`, `softplus`
- `swish` (custom), `sine`
- `elu`, `sel`, `elist`

Example structure:

```text
Input (1) → [Hidden x N] → Output (1)
```

---

## Usage

Clone the repository and install dependencies:

```bash
git clone https://github.com/hungtrannam/Physics-Informed-Neural-for-damped-oscillation.git
cd Physics-Informed-Neural-for-damped-oscillation
pip install -r requirements.txt
```

Run training:

```bash
python main/main_PINN2.py
```

Or use:

```bash
./script/main_PINN2.sh
```

Trained models, plots, and logs are saved in the `runs/` directory.

---

## Configurable Parameters

| Argument               | Description                                  |
|------------------------|----------------------------------------------|
| `--num_epochs`         | Number of training iterations                |
| `--learning_rate`      | Learning rate for Adam                       |
| `--num_neurons`        | Hidden layer size                            |
| `--num_hidden_layers`  | Number of hidden layers                      |
| `--dropout_rate`       | Dropout rate                                 |
| `--activation_name`    | Activation function                          |
| `--omega_eq`           | Weight for equation loss                     |
| `--omega_bc`           | Weight for boundary loss                     |
| `--omega_dt`           | Weight for data loss                         |
| `--seed`               | Random seed                                  |

---

## Outputs

- Training loss history (equation, boundary, data)
- Model predictions vs. exact solution
- Optional `.gif` animation of prediction over time
- Trained model weights saved as `.pth`
- Visualization and evaluation functions in `utils/`

---

## Author

**Hung Tran Nam**  
*Physics-Informed Machine Learning | PINNs | Neural ODEs*

---
