import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import src.eigenmode.solve_eigenv as se
import src.eigenmode.gen_M as gen_M

# Time evolution function, T(t) = A * cos(lambda * t) + B * sin(lambda * t)
def time_evolution_function(t, A=1, B=0, C=1, lambda_val=1, mode="cosine"):
    """
    calculate the time evolution function T(t) = A * cos(lambda * t) + B * sin(lambda * t)

    Parameters:
    - t (float): Time value
    - A (float): Amplitude for cosine term
    - B (float): Amplitude for sine term
    - lambda_val (float): Frequency parameter
    - mode (str): Mode of the time evolution function ('cosine', 'sine', 'damped_cosine')
    """
    if mode == "cosine":
        return A * np.cos(C * lambda_val * t) + B * np.sin(C * lambda_val * t)
    elif mode == "sine":
        return A * np.sin(C * lambda_val * t) + B * np.cos(C * lambda_val * t)
    elif mode == "damped_cosine":
        return np.exp(-0.1 * t) * (A * np.cos(C * lambda_val * t) + B * np.sin(C * lambda_val * t))
    else:
        raise ValueError("Unknown mode. Choose from 'cosine', 'sine', 'damped_cosine'.")

# Compute the time-space evolution solution U(x, y, t)
def compute_U(M, method="eigs", num_eigenvalues=1, shape_size=(5,5), time_mode="cosine", time_steps=100, T_max=10):
    eigenvalues, eigenvectors = se.solve_eigenproblem(M, method=method, num_eigenvalues=num_eigenvalues)
    
    # lambda = sqrt(-K)
    lambdas = np.sqrt(-eigenvalues)
    
    # time span
    t_values = np.linspace(0, T_max, time_steps)
    
    # initialize U(x, y, t)
    U = np.zeros((shape_size[0], shape_size[1], time_steps))
    
    for i, t in enumerate(t_values):
        T_t = time_evolution_function(t, A=1, B=1, lambda_val=lambdas[0], mode=time_mode)
        U[:, :, i] = eigenvectors[:, 0].reshape(shape_size) * T_t
    
    return U, t_values

# animation function
def animate_U(U, t_values, shape_size=(5,5)):
    """
    use matplotlib.animation to animate the time evolution of U(x, y, t)
    """
    fig, ax = plt.subplots()
    cax = ax.imshow(U[:, :, 0], cmap='coolwarm', vmin=-np.max(U), vmax=np.max(U))
    ax.set_title("Wave Equation Eigenmode Animation")
    
    def update(frame):
        cax.set_array(U[:, :, frame])
        ax.set_title(f"Time: {t_values[frame]:.2f}")
        return cax,
    
    ani = animation.FuncAnimation(fig, update, frames=len(t_values), interval=50, blit=False)
    plt.colorbar(cax)
    plt.show()
    
    return ani

if __name__ == "__main__":
    # Generate Laplacian matrix M for a square domain
    N = 50
    M = gen_M.generate_M_with_square(N, h=1/N)
    # Compute the time-space evolution solution U(x, y, t)
    U, t_values = compute_U(M, method="eigs", num_eigenvalues=1, shape_size=(N,N), time_mode="cosine", time_steps=100, T_max=10)
    # Animate the time evolution of U(x, y, t)
    animate_U(U, t_values, shape_size=(N,N))

    # circle domain
    M, mask = gen_M.generate_M_with_circle_v2(N, h=1/N)
    # Compute the time-space evolution solution U(x, y, t)
    U, t_values = compute_U(M, method="eigs", num_eigenvalues=1, shape_size=(N,N), time_mode="cosine", time_steps=100, T_max=10)
    # Animate the time evolution of U(x, y, t)
    animate_U(U, t_values, shape_size=(N,N))