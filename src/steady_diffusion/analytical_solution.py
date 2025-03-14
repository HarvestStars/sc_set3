import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def analytical_solution(N=50, L=2.0, source_x=0.6, source_y=1.2):
    """
    Solves the steady-state diffusion equation (Laplace equation) using a sparse linear system approach.
    
    Parameters:
    - N: int, number of grid points in each direction
    - L: float, domain size (square domain [-L, L] x [-L, L])
    - source_x: float, x-coordinate of the source point
    - source_y: float, y-coordinate of the source point
    
    Returns:
    - c_solution: 2D NumPy array, the computed concentration field
    - x, y: 1D NumPy arrays, the grid coordinates
    - mask: 2D NumPy array, boolean mask indicating the circular computational domain
    """
    
    # Grid setup
    h = 2 * L / (N - 1)  # Grid spacing
    x = np.linspace(-L, L, N)  # X-coordinates
    y = np.linspace(-L, L, N)  # Y-coordinates
    X, Y = np.meshgrid(x, y)  # Meshgrid for computation

    # Define the circular domain mask (radius = 2)
    mask = (X**2 + Y**2) <= 4  

    # Convert source point to grid indices
    i_s = np.argmin(np.abs(x - source_x))
    j_s = np.argmin(np.abs(y - source_y))

    # Construct sparse linear system Ax = b
    size = N * N  # Total number of unknowns
    A = lil_matrix((size, size))  # Sparse coefficient matrix
    b = np.zeros(size)  # Right-hand side vector

    # Loop through the grid and construct the linear system
    for i in range(N):
        for j in range(N):
            index = i * N + j  # Flattened 1D index for 2D (i, j)

            if not mask[i, j]:  # Boundary condition: c = 0 outside the circular domain
                A[index, index] = 1 # for stability
                b[index] = 0
                continue

            if i == i_s and j == j_s:  # Source point condition
                A[index, index] = 1
                b[index] = 1
                continue

            # Internal points - 5-point stencil for Laplace equation
            A[index, index] = -4  # Central coefficient
            if i > 0:
                A[index, (i-1) * N + j] = 1  # c[i-1, j]
            if i < N-1:
                A[index, (i+1) * N + j] = 1  # c[i+1, j]
            if j > 0:
                A[index, i * N + (j-1)] = 1  # c[i, j-1]
            if j < N-1:
                A[index, i * N + (j+1)] = 1  # c[i, j+1]

    # Convert A to CSR format for efficient solving
    A = A.tocsr()

    # Solve the linear system Ax = b
    c_solution = spsolve(A, b)

    # Reshape the solution into 2D grid form
    c_solution = c_solution.reshape((N, N))

    return c_solution, x, y


def plot_solution(c_solution, N, x, y, path="../../fig/steady_diffusion_circle_by_MatrixM.png"):
    """
    Plots the steady-state diffusion solution with a circular boundary.

    Parameters:
    - c_solution: 2D NumPy array, computed concentration field
    - x, y: 1D NumPy arrays, grid coordinates
    - mask: 2D NumPy array, boolean mask for the computational domain
    - source_x: float, x-coordinate of the source point
    - source_y: float, y-coordinate of the source point
    """

    # plot the concentration field
    fig, ax = plt.subplots(figsize=(6,5))
    plt.contourf(x, y, c_solution.T, levels=50, cmap="hot")
    plt.colorbar()

    # plot the circle boudary
    circle = plt.Circle((0, 0), 2, color='purple', fill=False, linewidth=2, linestyle="dashed")
    ax.add_patch(circle)

    # add note for the circle boundary
    ax.text(0, -1.8, "Purple dashed line: Circular boundary (for indication only)", 
            fontsize=8, color='white', ha='center', va='center', bbox=dict(facecolor='purple', alpha=0.5))

    # plt.scatter(source_x, source_y, color='blue', marker='o', label="Source Point (1)")
    plt.title(f"Concentration by Matrix M Analytical Method\nin Circular disk domain with radius=2, N={N}", fontsize=12, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("x", fontsize=14, fontweight='bold')
    plt.ylabel("y", fontsize=14, fontweight='bold')
    plt.savefig(path, dpi=300)
    plt.show()


if __name__ == "__main__":
    c_solution, x, y = analytical_solution()
    plot_solution(c_solution, 50, x, y)
