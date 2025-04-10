import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from utils.data_provider import GAMMA, OMEGA, t_data, u_data
from utils.config import get_args, set_seed
import numpy as np
import matplotlib.pyplot as plt

# 1. Định nghĩa hệ phương trình dao động tắt dần
def damped_oscillation(t, y):
    return np.array([y[1], -2 * GAMMA * y[1] - OMEGA**2 * y[0]])

# 2. Các phương pháp giải ODE

def euler(f, t0, y0, h, n):
    t_values, y_values = [t0], [y0]
    for _ in range(n):
        y_next = y_values[-1] + h * f(t_values[-1], y_values[-1])
        t_values.append(t_values[-1] + h)
        y_values.append(y_next)
    return np.array(t_values), np.array(y_values)

def improved_euler(f, t0, y0, h, n):  # Heun's method
    t_values, y_values = [t0], [y0]
    for _ in range(n):
        y_pred = y_values[-1] + h * f(t_values[-1], y_values[-1])
        y_corr = y_values[-1] + (h / 2) * (f(t_values[-1], y_values[-1]) + f(t_values[-1] + h, y_pred))
        t_values.append(t_values[-1] + h)
        y_values.append(y_corr)
    return np.array(t_values), np.array(y_values)

def runge_kutta_2(f, t0, y0, h, n):  # Midpoint method
    t_values, y_values = [t0], [y0]
    for _ in range(n):
        k1 = h * f(t_values[-1], y_values[-1])
        k2 = h * f(t_values[-1] + h / 2, y_values[-1] + k1 / 2)
        y_next = y_values[-1] + k2
        t_values.append(t_values[-1] + h)
        y_values.append(y_next)
    return np.array(t_values), np.array(y_values)

def runge_kutta_4(f, t0, y0, h, n):
    t_values, y_values = [t0], [y0]
    for _ in range(n):
        k1 = h * f(t_values[-1], y_values[-1])
        k2 = h * f(t_values[-1] + h/2, y_values[-1] + k1/2)
        k3 = h * f(t_values[-1] + h/2, y_values[-1] + k2/2)
        k4 = h * f(t_values[-1] + h, y_values[-1] + k3)
        y_next = y_values[-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        t_values.append(t_values[-1] + h)
        y_values.append(y_next)
    return np.array(t_values), np.array(y_values)

# 3. Liệt kê các phương pháp
methods = {
    'Euler': euler,
    'Improved Euler': improved_euler,
    'Runge Kutta 2': runge_kutta_2,
    'Runge Kutta 4': runge_kutta_4
}

# 4. Hàm chính giải bài toán và vẽ kết quả
def solver(method_name):
    t0 = 0
    y0 = np.array([1.0, 0.0])  # u(0) = 1, u'(0) = 0
    h = 0.1
    n = 150

    method = methods[method_name]
    t_values, y_values = method(damped_oscillation, t0, y0, h, n)

    y_exact = np.exp(-GAMMA * t_values) * np.cos(OMEGA * t_values)
    mse = np.mean((y_values[:, 0] - y_exact)**2)

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, y_exact, label='Exact Solution', color='red', linestyle='dashed')
    plt.plot(t_values, y_values[:, 0], label=f'{method_name} Method', color='blue', linewidth=2)
    plt.xlabel('Time $t$')
    plt.ylabel('Displacement $u(t)$')
    plt.title(f'Damped Oscillation: {method_name} (MSE: {mse:.6f})')
    plt.legend()
    plt.ylim(-1, 1)

    # Tạo thư mục lưu kết quả
    os.makedirs('./runs', exist_ok=True)
    plt.savefig(f'./runs/{method_name}_solution.pdf')
    plt.close()

    print(f'MSE ({method_name}): {mse:.6f}')
    print(f'Result saved to ./runs/{method_name}_solution.pdf')

# 5. Entry point
def main():
    parser = argparse.ArgumentParser(description='Solve damped oscillation using numerical methods.')
    parser.add_argument('--method', type=str, choices=methods.keys(), default='Runge Kutta 4', help='Numerical method to use')
    args = parser.parse_args()

    set_seed(42)
    print(f'🔧 Solving damped oscillation using {args.method.upper()} method...')
    solver(args.method)
    print('✅ Solution complete!')

if __name__ == "__main__":
    main()
