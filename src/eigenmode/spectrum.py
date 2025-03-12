import numpy as np
import matplotlib.pyplot as plt
import gen_M
import solve_egenv

def compute_eigenfrequency_spectrum(L_values, N, shape, num_eigenvalues=10, method="eigh"):
    """
    Compute the eigenfrequency spectrum for different values of L, given a fixed grid size N.
    
    Parameters:
    - L_values: List of L values to evaluate.
    - N: Grid size for discretization.
    - shape: The shape of the domain ('square', 'rectangle', 'circle').
    - num_eigenvalues: Number of smallest eigenvalues to compute.
    - method: Method to use for solving the eigenvalue problem ('eigh' by default).
    
    Returns:
    - A dictionary with L as keys and corresponding eigenfrequency arrays as values.
    """
    spectrum = {}
    
    for L in L_values:
        h = L / N  # Compute grid spacing

        if shape == 'square':
            M = gen_M.generate_M_with_square(N, h)  # Generate Laplacian matrix for square domain
        elif shape == 'rectangle':
            M = gen_M.generate_M_with_rectangle(N, h)
        elif shape == 'circle':
            M, _ = gen_M.generate_M_with_circle_v2(N, h) # v1 is not csr format, so use v2

        eigenvalues, eigenmodes = solve_egenv.solve_eigenproblem(M, method=method, num_eigenvalues=num_eigenvalues)
        eigenfrequencies = np.sqrt(-eigenvalues)  # Compute eigenfrequencies
        spectrum[L] = eigenfrequencies  # Store results
    
    return spectrum

# Visualizing the spectrum:
def plot_eigenfrequency_spectrum(spectrum):
    """
    Plot the eigenfrequency spectrum for different L values.
    
    Parameters:
    - spectrum: Dictionary where keys are L values and values are lists of eigenfrequencies.
    """
    plt.figure(figsize=(8, 6))
    for L, freqs in spectrum.items():
        plt.plot(range(1, len(freqs) + 1), freqs, marker='o', label=f"L={L}")
    
    plt.xlabel("Eigenmode Index")
    plt.ylabel("Eigenfrequency")
    plt.title("Eigenfrequency Spectrum for Different L Values")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_eigenfrequency_spectrum_by_shape(spectrum, shape):
    """
    Plot the eigenfrequency spectrum for different L values for a given shape.
    
    Parameters:
    - spectrum: Dictionary where keys are L values and values are lists of eigenfrequencies.
    - shape: The shape of the domain ('square', 'rectangle', 'circle').
    """
    plt.figure(figsize=(8, 6))

    for L, freqs in spectrum.items():
        L_array = np.full(len(freqs), L)  # Create an array of L values for plotting
        plt.scatter(L_array, freqs, label=f"L={L}", alpha=0.6)  # Scatter plot for each L

    plt.xlabel("Domain Size L")
    plt.ylabel("Eigenfrequency")
    plt.title(f"Eigenfrequency Spectrum for {shape.capitalize()} Shape")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage:
    L_values = [1, 2, 3, 4, 5]  # Different domain sizes
    N = 20  # Grid size
    shape = 'square'  # shape = 'rectangle' or 'circle'

    spectrum = compute_eigenfrequency_spectrum(L_values, N, "circle", num_eigenvalues=10, method="eigh")
    plot_eigenfrequency_spectrum_by_shape(spectrum, shape)