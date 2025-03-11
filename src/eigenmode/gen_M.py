import numpy as np
import scipy.sparse as sp

# v1,1 v1,2 v1,3
# v2,1 v2,2 v2,3
# v3,1 v3,2 v3,3
# N = 3
# N^2 x N^2 grid, here is 9 x 9 grid

def generate_M_with_square(N, h=1.0):
    """
    Generate the discretized Laplacian matrix M for a square computational domain
    with Dirichlet boundary conditions (boundary values set to zero).
    
    Parameters:
        N (int): Number of interior grid points in one dimension (excluding boundary points).
        h (float): Grid spacing

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix M of shape (N^2, N^2) representing the Laplacian operator.
    """
    # Total number of interior points
    num_points = N * N
    
    # Create sparse matrix with five-point stencil
    main_diag = -4 / h**2 * np.ones(num_points)  # Center coefficient (-4)
    side_diag = 1/ h**2 * np.ones(num_points - 1)   # Left and right neighbors (+1)
    up_down_diag = 1/ h**2 * np.ones(num_points - N) # Up and down neighbors (+1)

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
def generate_M_with_rectangle(N, h=1.0):
    """
    Generate the discretized Laplacian matrix M for a rectangular computational domain
    with Dirichlet boundary conditions (boundary values set to zero).

    Parameters:
        N (int): Number of interior grid points in one dimension (excluding boundary points).
        h (float): Grid spacing

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix M of shape (N * 2N, N * 2N) representing the Laplacian operator.
    """
    Nx, Ny = 2*N, N  # Nx: long side, Ny: short side
    num_points = Nx * Ny  # Total number of interior points

    # center diag line: coefficient (-4)
    main_diag = -4 / h**2 * np.ones(num_points)

    # left and right neighbors diag line: (+1)
    side_diag = 1/ h**2 * np.ones(num_points - 1)

    # disconnect right side of each row
    for i in range(1, Nx):
        side_diag[i * Ny - 1] = 0

    # up and down neighbors diag line: (+1)
    up_down_diag = 1 / h**2 * np.ones(num_points - Ny)

    # Create sparse matrix using diagonals
    M = sp.diags(
        [main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
        [0, -1, 1, -Ny, Ny],  # Diagonal offsets
        shape=(num_points, num_points),
        format="csr"
    )

    return M

def generate_M_with_circle_v1(N, h=1.0):
    """
    Generate a sparse matrix M (N x N) corresponding to the discrete Laplacian matrix for a circular computational domain.
    Uses a 5-point finite difference scheme with Dirichlet boundary conditions (boundary values set to 0).

    Parameters:
        N (int): Grid size of the computational domain (NxN)
        h (float): Grid spacing

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix M, with shape (N*N, N*N)
        index_map (dict): Mapping from 2D coordinates to the corresponding index in the flattened matrix
        valid_points (numpy.array): 2D coordinates of points inside the circle
    """
    R = N // 2  # radius of the circle (assuming diameter = grid size)

    # generate index grid
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    x, y = x - R, y - R  # 以中心 (0,0) 对齐

    # generate circular mask
    mask = (x**2 + y**2 <= R**2)

    # get valid points inside the circle
    valid_points = np.argwhere(mask) # coordinates of points inside the circle
    num_valid = len(valid_points)  # number of valid points

    # create index map
    index_map = {tuple(pt): idx for idx, pt in enumerate(valid_points)}

    # create sparse matrix M
    M = sp.lil_matrix((num_valid, num_valid))

    # fill the matrix
    for idx, (i, j) in enumerate(valid_points):
        M[idx, idx] = -4 / h**2 # diagonal element
        
        # left, right, up, down
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if (ni, nj) in index_map:  # if the neighbor is inside the circle
                M[idx, index_map[(ni, nj)]] = 1 / h**2

    return M.tocsr(), index_map, valid_points


def generate_M_with_circle_v2(N, h=1.0):
    """
    Generate a sparse matrix M (N x N) corresponding to the discrete Laplacian matrix for a circular computational domain.
    Uses a 5-point finite difference scheme with Dirichlet boundary conditions (boundary values set to 0).
    
    Parameters:
        N (int): Grid size of the computational domain (NxN)
        h (float): Grid spacing
    
    Returns:
        scipy.sparse.csr_matrix: Sparse matrix M, with shape (N*N, N*N)
        mask (numpy.array): 1D array marking which points are inside (1) / outside (0) the circle
    """
    R = N // 2  # Radius of the circle (assuming diameter = grid size)
    num_points = N * N  # Total number of grid points
    
    # Create index grid
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    x = x - R  # Center at (0,0)
    y = y - R

    # Create circular mask (1 = inside the circle, 0 = outside)
    mask = (x**2 + y**2) <= R**2
    mask_flat = mask.flatten()  # Flatten to 1D

    # Construct five-point finite difference matrix
    main_diag = -4 / h**2 * np.ones(num_points)
    side_diag = 1 / h**2 * np.ones(num_points - 1)
    up_down_diag = 1 / h**2 * np.ones(num_points - N)

    # Handle row boundary conditions (prevent incorrect connections)
    for i in range(1, N):
        side_diag[i * N - 1] = 0

    # Generate standard Laplacian matrix
    M = sp.diags(
        [main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
        [0, -1, 1, -N, N], 
        shape=(num_points, num_points),
        format="csr"
    )

    # **Process points outside the circle: set corresponding rows to zero**
    for i in range(num_points):
        if not mask_flat[i]:  # If the point is outside the circle
            M[i, :] = 0  # Set the row to zero
            M[i, i] = 1 / h**2  # Ensure numerical stability (nonzero diagonal)

    return M, mask_flat



if __name__ == "__main__":
    N = 5  # Example for a 3x3 interior grid (excluding boundary)
    M_square = generate_M_with_square(N - 2) # Generate M for a 3x3 interior grid
    print(M_square.toarray())  # Display the full matrix for small N

    M_rectangle = generate_M_with_rectangle(N - 2)
    print(M_rectangle.toarray())

    M_circle, index_map, valid_points = generate_M_with_circle_v1(N)
    print(f"Matrix size: {M_circle.shape}") 

    M_circle, mask = generate_M_with_circle_v2(N)
    print(M_circle.toarray()) 
