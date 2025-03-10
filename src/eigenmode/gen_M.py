import numpy as np
import scipy.sparse as sp

# v1,1 v1,2 v1,3
# v2,1 v2,2 v2,3
# v3,1 v3,2 v3,3
# N = 3
# N^2 x N^2 grid, here is 9 x 9 grid

def generate_M_with_square(N):
    """
    Generate the discretized Laplacian matrix M for a square computational domain
    with Dirichlet boundary conditions (boundary values set to zero).
    
    Parameters:
        N (int): Number of interior grid points in one dimension (excluding boundary points).
    
    Returns:
        scipy.sparse.csr_matrix: Sparse matrix M of shape (N^2, N^2) representing the Laplacian operator.
    """
    # Total number of interior points
    num_points = N * N
    
    # Create sparse matrix with five-point stencil
    main_diag = -4 * np.ones(num_points)  # Center coefficient (-4)
    side_diag = np.ones(num_points - 1)   # Left and right neighbors (+1)
    up_down_diag = np.ones(num_points - N) # Up and down neighbors (+1)

    # Adjust side diagonals to ensure correctness at boundaries
    for i in range(1, N):
        side_diag[i * N - 1] = 0  # Set to zero at right boundary

    # Create sparse matrix using diagonals
    M = sp.diags(
        [main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
        [0, -1, 1, -N, N], # Diagonal offsets
        shape=(num_points, num_points),
        format="csr"
    )

    return M

# v1,1 v1,2 v1,3
# v2,1 v2,2 v2,3
# v3,1 v3,2 v3,3
# v4,1 v4,2 v4,3
# v5,1 v5,2 v5,3
# v6,1 v6,2 v6,3
# N=3, 2N=6
# N*2N x N*2N grid
def generate_M_with_rectangle(N):
    """
    Generate the discretized Laplacian matrix M for a rectangular computational domain
    with Dirichlet boundary conditions (boundary values set to zero).

    Parameters:
        N (int): Number of interior grid points in one dimension (excluding boundary points).

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix M of shape (N * 2N, N * 2N) representing the Laplacian operator.
    """
    Nx, Ny = 2*N, N  # Nx: long side, Ny: short side
    num_points = Nx * Ny  # Total number of interior points

    # center diag line: coefficient (-4)
    main_diag = -4 * np.ones(num_points)

    # left and right neighbors diag line: (+1)
    side_diag = np.ones(num_points - 1)

    # disconnect right side of each row
    for i in range(1, Nx):
        side_diag[i * Ny - 1] = 0

    # up and down neighbors diag line: (+1)
    up_down_diag = np.ones(num_points - Ny)

    # Create sparse matrix using diagonals
    M = sp.diags(
        [main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
        [0, -1, 1, -Ny, Ny],  # Diagonal offsets
        shape=(num_points, num_points),
        format="csr"
    )

    return M

# Example usage:
N = 5  # Example for a 3x3 interior grid (excluding boundary)
M_square = generate_M_with_square(N - 2) # Generate M for a 3x3 interior grid
print(M_square.toarray())  # Display the full matrix for small N

M_rectangle = generate_M_with_rectangle(N - 2)
print(M_rectangle.toarray())