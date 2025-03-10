import gen_M
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt

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
        w, v = spla.eigs(M, k=num_eigenvalues, which='SM')  # 'SM' means computing the smallest eigenvalues
        return w.real, v.real  # Ensure the result is real

    else:
        raise ValueError("Invalid method. Choose from 'eig', 'eigh', or 'eigs'.")

def plot_eigenmodes(M, N, num_modes=3):
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
    fig, axes = plt.subplots(1, num_modes, figsize=(4 * num_modes, 4))
    
    for i in range(num_modes):
        eigenvector = v[:, i].reshape(2 * N, N)  # Reshape the eigenvector to a 2D grid

        ax = axes[i]
        im = ax.imshow(eigenvector, cmap='coolwarm', origin='lower', aspect='auto')
        ax.set_title(f"Eigenmode {i+1}\nλ={w[i]:.4f}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

# # Example: Solve the eigenproblem for M matrix on a rectangular computational domain
# N = 5  # Grid size for the rectangle
# M_rectangle = gen_M.generate_M_with_square(N)  # Generate M matrix

# # Choose an appropriate solution method
# w, v = solve_eigenproblem(M_rectangle, method="eigs", num_eigenvalues=6)

# # Print the smallest 6 eigenvalues
# print("Smallest eigenvalues:", w)

# Example usage
N = 100  # Short side of the computational domain
M_rectangle = gen_M.generate_M_with_rectangle(N)  # Generate M for the rectangular domain
plot_eigenmodes(M_rectangle, N, num_modes=3)  # Plot the first three eigenvector heatmaps
