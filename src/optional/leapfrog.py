import numpy as np
from numba import njit 
from scipy.integrate import RK45

# Spring
# dx/dt = v_i
# dv_i/dt = Fi({x})/m_i = - dU({x})/dx_i

# note = pos only depends on the vel and vice verse (no self-dependence)
# calculate the position and velocity a half step apart. x=n & v=n+1/2

# fwd method:

# region Leapfrog stuff
@njit
def leapStepOscillator(x_i, v_i_half, dt, k):
    
    # mass is 1 and left out
    x_i_1 = x_i + v_i_half * dt # EQ 27
    
    v_i_threehalf = v_i_half + (-k)*x_i_1*dt # EQ 28
    
    v_i = v_i_half + (-k)*x_i_1*dt*0.5 # used for energy calc and plotting
    
    return x_i_1, v_i_threehalf, v_i

@njit
def oscillateLeap(N, T, k=1, x0=1, v0=None):
    
    t = np.linspace(0, T, N)
    dt = t[1] - t[0]
    
    x_n = np.zeros(N)
    v_n_half = np.zeros(N)
    e = np.zeros(N)
    v_n = np.zeros(N)
    
    x_n[0] = x0
    # if v0 is not set calculate it from first position
    if v0 is None:
        v_n_half[0] = (-k * x_n[0]) * dt * 0.5 # perform a half step to get first v value
    else:
        v_n_half[0] = v0
    
    for i in range(N-1):
        x_n[i + 1], v_n_half[i + 1], v_n[i+1] = leapStepOscillator(x_n[i], v_n_half[i], dt, k)
        
        e[i+1] = 0.5 * (k*x_n[i+1]**2 + v_n[i+1]**2)
        
    # last energy calculation not performed
    e[0] = 0.5 * (k*x_n[0]**2 + v_n[0]**2)
    e[-1] = e[-2]
    return x_n, v_n, e
    
# region RK45 stuff
def harmonic_oscillator(t, y, k):
    x, v = y
    dxdt = v
    dvdt = -k*x # m = 1 so acceleration is -k*x / 1
    return [dxdt, dvdt]

def oscillateRK45(N, T, k=1, x0=1, v0 = 0):

    y0 = [x0, v0]
    max_step = T/(N-1)
    solver = RK45(fun = lambda t, y: harmonic_oscillator(t, y, k),
                  t0 = 0, y0 = y0, t_bound=T, max_step=max_step)
    
    t_values = [solver.t]
    x_values = [solver.y[0]]
    v_values = [solver.y[1]]

    while solver.status == 'running':
        solver.step()
        t_values.append(solver.t)
        x_values.append(solver.y[0])
        v_values.append(solver.y[1])
    
    e_values = []
    for i in range(len(x_values) - 1):
        e = 0.5 * (k*x_values[i]**2 + (v_values[i])**2)
        e_values.append(e)
    e_values.append(0.5 * (k*x_values[-1]**2 + v_values[-1]**2))
    
    return t_values, x_values, v_values, e_values
    
# region forced

@njit
def leapStepForcedOscillator(x_i, v_i_half, dt, t, strength, frequency, k):
    x_i_1 = x_i + v_i_half * dt
    # natural frequency is sqrt(k)
    v_i_threehalf = v_i_half + (strength*np.sin(frequency*t) + (-k)*x_i_1)*dt # EQ 29
    
    v_i = v_i_half + (-k)*x_i_1*dt*0.5
    
    return x_i_1, v_i_threehalf, v_i

@njit
def oscillateForcedLeap(N, T, strength, frequency, k=1, x0=1, v0=None):
    
    t = np.linspace(0, T, N)
    dt = t[1] - t[0]
    
    x_n = np.zeros(N)
    v_n_half = np.zeros(N)
    e = np.zeros(N)
    v_n = np.zeros(N)
    
    x_n[0] = x0
    # if v0 is not set calculate it from first position
    if v0 is None:
        v_n_half[0] = (-k * x_n[0]) * dt * 0.5 # perform a half step to get first v value
    else:
        v_n_half[0] = v0
    
    for i in range(N-1):
        x_n[i + 1], v_n_half[i + 1], v_n[i+1] = leapStepForcedOscillator(x_n[i], v_n_half[i], dt, t[i], strength, frequency, k)
        
        e[i+1] = 0.5 * (k*x_n[i+1]**2 + v_n[i+1]**2)
        
    # last energy calculation not performed
    e[0] = 0.5 * (k*x_n[0]**2 + v_n[0]**2)
    e[-1] = e[-2]
    return x_n, v_n, e
# region Main
if __name__ == "__main__":
    N = 10000
    T = 100
    k_values = [0.5, 1, 1.5]
    
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for idx, k in enumerate(k_values):
        position, velocity, energy = oscillateLeap(N, T, k=k)
        
        t = np.linspace(0, T, int(N))
        
        axs[0].plot(t, position, label=f'k={k}')
        axs[0].set_ylabel('Position')
        axs[0].legend()

        axs[1].plot(t, velocity, label=f'k={k}')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()

        axs[2].plot(t, energy, label=f'k={k}')
        axs[2].set_ylabel('Energy')
        axs[2].set_xlabel('Time')
        axs[2].legend()

    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for idx, k in enumerate(k_values):
        t, position, velocity, energy = oscillateRK45(N, T, k=k)
        
        axs[0].plot(t, position, label=f'k={k}')
        axs[0].set_ylabel('Position')
        axs[0].legend()

        axs[1].plot(t, velocity, label=f'k={k}')
        axs[1].set_ylabel('Velocity')
        axs[1].legend()

        axs[2].plot(t, energy, label=f'k={k}')
        axs[2].set_ylabel('Energy')
        axs[2].set_xlabel('Time')
        axs[2].legend()

    plt.tight_layout()
    plt.show()