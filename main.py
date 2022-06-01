from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

A_0 = 1
L = 1
K = 1

s = 0.15
dep = 0.05
A = A_0
g = 0.02  # double check; seems sus
theta = 0.1
gamma = 0.2
ro = 1.0
alpha = 0.03  # parameter for Y

simulation_time = 20
t_delta = 0.01
steps = int(simulation_time / t_delta)
t = 0

assert ro > gamma and gamma > 0
assert theta > 0


def L_der_div_L(t):
    return theta * gamma * np.exp(-theta * t) / (ro - gamma*np.exp(-theta * t))

def update_L(Y, K, A, L, t, t_delta):
    L_der = L_der_div_L(t) * L
    return t_delta * L_der + L


def update_K(Y, K, A, L, t, t_delta):
    K_der = s*Y - dep * K
    return t_delta * K_der + K

def update_Y(K, A, L):
    return K**alpha*(A*L)**(1-alpha)

def update_A(Y, K, A, L, t, t_delta):
    return A_0 * np.exp(g*t)


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
    param_history['k'].append(K/L)
    param_history['y'].append(Y/L)
    param_history['K'].append(K)
    param_history['Y'].append(Y)
    param_history['A'].append(A)
    param_history['L'].append(L)

    times.append(t)

    t += t_delta


for name, history in param_history.items():
    plt.plot(times, history)
    plt.title(name)
    plt.show()
