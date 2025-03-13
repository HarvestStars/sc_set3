import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

import src.eigenmode.gen_M as gen_M
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np

# Used for statistical analysis for different methods
def solve_eigenproblem(M, method="eigs", num_eigenvalues=6):
    """
    Compute the eigenvalues and eigenvectors of matrix M.
    
    Parameters:
    M (scipy.sparse matrix or numpy array): The matrix to be solved (N^2, N^2)
    method (str): The method to use, choose from "eig", "eigh", or "eigs"
    num_eigenvalues (int): If using "eigs", specify the number of eigenvalues to compute
    
    Returns:
    w (numpy array): Computed eigenvalues
    v (numpy array): Computed eigenvectors
    """
    if method == "eig":
        print("Using scipy.linalg.eig() (dense matrix solver)")
        M_dense = M.toarray() if hasattr(M, "toarray") else M
        w, v = la.eig(M_dense)  # Suitable for general matrices
        return w.real, v.real  # Ensure results are real
    
    elif method == "eigh":
        print("Using scipy.linalg.eigh() (for symmetric dense matrices)")
        M_dense = M.toarray() if hasattr(M, "toarray") else M
        w, v = la.eigh(M_dense)  # Suitable for symmetric matrices
        return w, v
    
    elif method == "eigs":
        print(f"Using scipy.sparse.linalg.eigs() (for sparse matrices, computing {num_eigenvalues} smallest eigenvalues)")
        M_dense = M.toarray() if hasattr(M, "toarray") else M
        w, v = spla.eigs(M, k=num_eigenvalues, which='SM')  # 'SM' means computing the smallest eigenvalues
        return w.real, v.real  # Ensure the result is real

    else:
        raise ValueError("Invalid method. Choose from 'eig', 'eigh', or 'eigs'.")

def plot_eigenmodes(M, reshape_x, reshape_y, num_modes=3, path="../../fig/eigenmodes.png"):
    """
    Compute and visualize the heatmaps of eigenvectors corresponding to the smallest eigenvalues of Laplacian matrix M.
    
    Parameters:
    M (scipy.sparse matrix): Laplacian matrix
    N (int): Number of grid points on the short side of the computational domain (rectangular size is (2N, N))
    num_modes (int): Number of eigenvectors to plot corresponding to the smallest eigenvalues
    
    Returns:
    None (directly generates the plots)
    """
    # Compute the first num_modes smallest eigenvalues and eigenvectors
    w, v = spla.eigs(M, k=num_modes, which='SM')  # 'SM' -> Smallest Magnitude
    w, v = w.real, v.real
    
    # Plot the eigenvectors as heatmaps
    shape = "Rectangle" if reshape_x != reshape_y else "Square"
    fig, axes = plt.subplots(1, num_modes, figsize=(4 * num_modes, 4))
    
    for i in range(num_modes):
        eigenvector = v[:, i].reshape(reshape_x, reshape_y)  # Reshape the eigenvector to a 2D grid
        ax = axes[i]
        im = ax.imshow(eigenvector, cmap='coolwarm', origin='lower', aspect='auto')
        ax.set_title(f"Shape {shape} Eigenmode {i+1}\nK={w[i]:.4f}, eigen_freq λ={np.sqrt(-w[i]):.4}") # λ**2 = -K
        ax.set_xlabel("X-axis", fontsize=16)  # adjust the X-axis label font size
        ax.set_ylabel("Y-axis", fontsize=16)

        # adjust the font size of the scale
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()

def solve_eigenproblem_circle_v1(M, index_map, N, num_modes=3, path="../../fig/eigenmodes_circle.png"):
    """
    Compute and visualize the heatmaps of eigenvectors corresponding to the smallest eigenvalues of Laplacian matrix M.

    Parameters:
    M (scipy.sparse matrix): Laplacian matrix
    index_map (dict): Mapping (i, j) -> index for solving the eigenvalues to map back to the grid
    N (int): Grid size of the computational domain (NxN)
    num_modes (int): Number of eigenvectors to plot corresponding to the smallest eigenvalues

    Returns:
    None (directly generates the plots)

    """

    # calculate the first num_modes smallest eigenvalues and eigenvectors
    w, v = spla.eigs(M, k=num_modes, which='SM')  # 'SM' -> Smallest Magnitude
    w, v = w.real, v.real

    # plot the eigenvectors as heatmaps
    fig, axes = plt.subplots(1, num_modes, figsize=(4 * num_modes, 4))
    
    for i in range(num_modes):
        eigenvector = np.zeros((N, N))  # initialize with zeros
        
        # fill the eigenvector with the inside points
        for (x, y), idx in index_map.items():
            eigenvector[x, y] = v[idx, i]  # fill the value of the eigenvector

        # plot the eigenvector
        ax = axes[i]
        im = ax.imshow(eigenvector, cmap='coolwarm', origin='lower', aspect='auto')
        ax.set_title(f"Shape Circle Eigenmode {i+1}\nK={w[i]:.4f}, eigen_freq λ={np.sqrt(-w[i]):.4}")
        ax.set_xlabel("X-axis", fontsize=14)  # adjust the X-axis label font size
        ax.set_ylabel("Y-axis", fontsize=14)

        # adjust the font size of the scale
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()

def solve_eigenproblem_circle_v2(M, mask, N, num_modes=3, path="../../fig/eigenmodes_circle_v2.png"):
    """
    Compute and visualize the heatmaps of eigenvectors corresponding to the smallest eigenvalues of Laplacian matrix M.
    
    Parameters:
    M (scipy.sparse matrix): Laplacian matrix
    mask (numpy.array): 1D array marking which points are inside (1) / outside (0) the circle
    N (int): Grid size of the computational domain (NxN)
    num_modes (int): Number of eigenvectors to plot corresponding to the smallest eigenvalues
    
    Returns:
    None (directly generates the plots)
    """
    # Compute the first num_modes smallest eigenvalues and eigenvectors
    w, v = spla.eigs(M, k=num_modes, which='SM')
    w, v = w.real, v.real  # Extract real parts

    print(v.shape)

    # Generate plots
    fig, axes = plt.subplots(1, num_modes, figsize=(4 * num_modes, 4))
    
    for i in range(num_modes):
        eigenvector = np.zeros(N * N)  # Initialize with zeros
        for j in range(N * N):
            if mask[j]:  # Fill only points inside the circle
                eigenvector[j] = v[j, i]
        eigenvector = eigenvector.reshape(N, N)  # Reshape to 2D

        ax = axes[i]
        im = ax.imshow(eigenvector, cmap='coolwarm', origin='lower', aspect='auto')
        ax.set_title(f"Shape Circle Eigenmode {i+1}\nK={w[i]:.4f}, eigen_freq λ={np.sqrt(-w[i]):.4}")
        ax.set_xlabel("X-axis", fontsize=14)  # adjust the X-axis label font size
        ax.set_ylabel("Y-axis", fontsize=14)

        # adjust the font size of the scale
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()


if __name__ == "__main__":
    # example usage
    N = 50  # Grid size for the circle
    h = 1/N  # Grid spacing

    # Square domain
    M_rectangle = gen_M.generate_M_with_square(N, h)  # Generate M matrix
    plot_eigenmodes(M_rectangle, N, N, num_modes=4)  # Plot the first three eig

    # Rectangular domain
    M_rectangle = gen_M.generate_M_with_rectangle(N, h)  # Generate M for the rectangular domain
    plot_eigenmodes(M_rectangle, 2*N, N, num_modes=4)  # Plot the first three eigenvector heatmaps

    # Circular domain
    M_circle, index_map, valid_points = gen_M.generate_M_with_circle_v1(N, h) 
    solve_eigenproblem_circle_v1(M_circle, index_map, N, num_modes=4)

    M_circle_v2, mask= gen_M.generate_M_with_circle_v2(N, h)
    solve_eigenproblem_circle_v2(M_circle_v2, mask, N, num_modes=4)

    # M_circle, mask = gen_M.generate_M_with_circle_v2(N)
    # solve_eigenproblem_circle_v2(M_circle, mask, N, num_modes=3)
