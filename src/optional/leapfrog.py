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
def leapStepOscillator(x_now, v_now, dt, k):
    
    # mass is 1 and left out
    #v_next = v_n+3/2
    # etc
    x_next = x_now + v_now*dt 
    v_next = v_now + (-k*x_next)*dt
    
    return x_next, v_next

@njit
def oscillateLeap(N, T, k=1, x0=1, v0=None):
    
    t = np.linspace(0, T, N)
    dt = t[1] - t[0]
    
    x = np.zeros(N)
    v = np.zeros(N)
    e = np.zeros(N)
    
    x[0] = x0
    # if v0 is not set calculate it from first position
    if v0 is None:
        v[0] = (-k * x[0]) * dt
    else:
        v[0] = v0
    
    for i in range(N-1):
        x[i + 1], v[i + 1] = leapStepOscillator(x[i], v[i], dt, k)
        e[i] = 0.5 * (k*x[i]**2 + ((v[i] + v[i + 1]) / 2)**2)
        
    e[-1] = 0.5 * (k*x[-1]**2 + v[-1]**2)
    return x, v, e
    
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
        e = 0.5 * (k*x_values[i]**2 + ((v_values[i] + v_values[i + 1]) / 2)**2)
        e_values.append(e)
    e_values.append(0.5 * (k*x_values[-1]**2 + v_values[-1]**2))
    
    return t_values, x_values, v_values, e_values
    

# region Main
if __name__ == "__main__":
    N = 10000
    T = 300
    k = 3
    position, velocity, energy = oscillateLeap(N, T, k=k)
    
    import matplotlib.pyplot as plt

    t = np.linspace(0, T, int(N))
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(t, position)
    axs[0].set_ylabel('Position')

    axs[1].plot(t, velocity)
    axs[1].set_ylabel('Velocity')

    axs[2].plot(t, energy)
    axs[2].set_ylabel('Energy')
    axs[2].set_xlabel('Time')

    plt.tight_layout()
    plt.show()
    
    t, position, velocity, energy = oscillateRK45(N, T, k=k)
    
    import matplotlib.pyplot as plt

    print(velocity[0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(t, position)
    axs[0].set_ylabel('Position')

    axs[1].plot(t, velocity)
    axs[1].set_ylabel('Velocity')

    axs[2].plot(t, energy)
    axs[2].set_ylabel('Energy')
    axs[2].set_xlabel('Time')

    plt.tight_layout()
    plt.show()