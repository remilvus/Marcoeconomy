from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import scipy.special

A_0 = 1
L = 1
K = 1

s = 0.15
d = 0.05
A = A_0
g = 0.02
theta = 0.1
gamma = 0.2
delta = 1.0
alpha = 0.03  # parameter for Y

simulation_time = 100
t_delta = 0.01
steps = int(simulation_time / t_delta)
t = 0

taylor_degree = 10  # how many elements of the taylor series are used

assert delta > gamma and gamma > 0
assert theta > 0


def L_der_div_L(t):
    return theta * gamma * np.exp(-theta * t) / (delta - gamma * np.exp(-theta * t))


def update_L(Y, K, A, L, t, t_delta):
    L_der = L_der_div_L(t) * L
    return t_delta * L_der + L


def update_K(Y, K, A, L, t, t_delta):
    K_der = s * Y - d * K
    return t_delta * K_der + K


def update_Y(K, A, L):
    return K**alpha * (A * L) ** (1 - alpha)


def update_A(Y, K, A, L, t, t_delta):
    return A_0 * np.exp(g * t)


beta = 1 - alpha


def calculate_c(t, taylor_degree):
    return (
        beta
        * s
        * A_0**beta
        * sum(
            [
                delta ** (beta - n)
                * (-1) ** n
                * scipy.special.binom(beta, n)
                * np.exp(t * ((g + d) * beta - theta * n))
                / ((g + d) * beta - theta * n)
                * gamma**n
                for n in range(taylor_degree + 1)
            ]
        )
    )


const = delta - gamma - calculate_c(0, taylor_degree)
print(f"C={const}")

times = [t_delta * i for i in range(steps)]
k_series = np.array(
    [
        (const + calculate_c(t, taylor_degree)) ** (1 / beta)
        / (np.exp(t * d) * (delta - gamma * np.exp(-theta * t)))
        for t in times
    ]
)
y_series = ...  # todo: calculate 'y' (this can be done directly from `k_series`)

param_history = defaultdict(list)
times = []
prev_A = A
prev_K = K
prev_L = L
prev_Y = update_Y(prev_K, prev_A, prev_L)

for i in range(steps):
    print(prev_L)
    print(prev_A)
    print(prev_Y)
    print(prev_K)
    print()

    Y = update_Y(prev_K, prev_A, prev_L)
    K = update_K(prev_Y, prev_K, prev_A, prev_L, t, t_delta)
    L = update_L(prev_Y, prev_K, prev_A, prev_L, t, t_delta)
    A = update_A(prev_Y, prev_K, prev_A, prev_L, t, t_delta)
    prev_A = A
    prev_Y = Y
    prev_K = K
    prev_L = L
    param_history["k"].append(K / L)
    param_history["y"].append(Y / L)
    param_history["K"].append(K)
    param_history["Y"].append(Y)
    param_history["A"].append(A)
    param_history["L"].append(L)

    times.append(t)

    t += t_delta


for name, history in param_history.items():
    plt.plot(times, history)
    plt.title(name + " (symulacja)")
    plt.show()

plt.plot(times, k_series)
plt.title("k (suma szeregu)")
plt.show()

k_simulation = np.array(param_history["k"])
plt.plot(times, k_series - k_simulation)
plt.title("k (różnica)")
plt.show()

# todo: add plots for 'y': "y (suma szeregu)" and "y (różnica)"
# todo: save all figures to use in LATEX
