import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

# Jacobi iteration with Numba
@jit(nopython=True, parallel=True)
def jacobi_iteration(c_old, c_new, source_index_x, source_index_y, mask, N, tol, max_iters=5000):
    error_arr = np.zeros((N, N))

    for _ in range(max_iters):
        error = 0
        for i in prange(1, N-1):
            for j in prange(1, N-1):
                if not mask[i, j] or (i == source_index_x and j == source_index_y):  # skip the boundary points and the source point
                    continue
                c_new[i, j] = 0.25 * (c_old[i+1, j] + c_old[i-1, j] + c_old[i, j+1] + c_old[i, j-1])
                error_arr[i, j] = abs(c_new[i, j] - c_old[i, j])
        
        c_old[:,:] = c_new[:,:]  # update c_old

        error = np.max(error_arr)
        if error < tol:
            break

def plot_circle_domain(N=50, R=2.0, tol=1e-6, source_point=(0.6, 1.2), path="../../fig/steady_diffusion_circle_by_jacobi.png"):
    # h = 2 * L / (N - 1)  # grid spacing
    # initialize concentration field
    c = np.zeros((N, N))
    x = np.linspace(-R, R, N)
    y = np.linspace(-R, R, N)
    X, Y = np.meshgrid(x, y)  # meshgrid for plotting

    mask = (X**2 + Y**2) <= R**2  # circular mask!!!
    # if all set 1, then the mask is a square
    # mask = np.ones((N, N), dtype=bool)  # entire domain

    # set source point
    source_x, source_y = source_point[0], source_point[1]
    
    # Row-wise stores x: along the rows direction, x is stored
    # Column-wise stores y. 
    # !!!
    # Pay attention to the indexing, in matplot, it interprets the first index as y, and the second index as x!!!
    # So, either you transpose the matrix c after interative or you need to switch the i_s and j_s here.
    i_s = np.argmin(np.abs(x - source_x)) 
    j_s = np.argmin(np.abs(y - source_y))
    c[i_s, j_s] = 1  # source point

    # solve the steady-state diffusion equation
    c_new = np.copy(c)
    jacobi_iteration(c, c_new, i_s, j_s, mask, N, tol)

    # plot the concentration field
    fig, ax = plt.subplots(figsize=(6,5))
    plt.contourf(x, y, c.T, levels=50, cmap="hot")
    plt.colorbar()

    # plot the circle boudary
    circle = plt.Circle((0, 0), 2, color='purple', fill=False, linewidth=2, linestyle="dashed")
    ax.add_patch(circle)

    # add note for the circle boundary
    ax.text(0, -1.8, "Purple dashed line: Circular boundary (for indication only)", 
            fontsize=8, color='white', ha='center', va='center', bbox=dict(facecolor='purple', alpha=0.5))

    # plt.scatter(source_x, source_y, color='blue', marker='o', label="Source Point (1)")
    plt.title(f"Concentration by Jacobi Iterative Method\nin Circular disk domain with radius=2, N={N}", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("x", fontsize=14, fontweight='bold')
    plt.ylabel("y", fontsize=14, fontweight='bold')
    # plt.legend()
    plt.savefig(path, dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_circle_domain()