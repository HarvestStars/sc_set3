import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

import numpy as np
import matplotlib.pyplot as plt
import src.eigenmode.gen_M as gen_M
import src.eigenmode.solve_eigenv as solve_eigenv

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

        eigenvalues, eigenmodes = solve_eigenv.solve_eigenproblem(M, method=method, num_eigenvalues=num_eigenvalues)
        eigenfrequencies = np.sqrt(-eigenvalues)  # Compute eigenfrequencies
        spectrum[L] = eigenfrequencies  # Store results
    
    return spectrum

# Visualizing the spectrum:
def plot_eigenfrequency_spectrum(spectrum, path=None):
    """
    Plot the eigenfrequency spectrum for different L values.
    
    Parameters:
    - spectrum: Dictionary where keys are L values and values are lists of eigenfrequencies.
    """
    plt.figure(figsize=(8, 6))
    for L, freqs in spectrum.items():
        plt.plot(range(1, len(freqs) + 1), freqs, marker='o', label=f"L={L}")
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Eigenmode Index", fontsize=14, fontweight='bold')
    plt.ylabel("Eigenfrequency", fontsize=14, fontweight='bold')
    plt.title("Eigenfrequency Spectrum for Different L Values", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(path)
    plt.show()

def plot_eigenfrequency_spectrum_by_shape(spectrum, shape, path="./fig/eigenfrequency_spectrum.png"):
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

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel("Domain Size L", fontsize=14, fontweight='bold')
    plt.ylabel("Eigenfrequencies(spectrum)", fontsize=14, fontweight='bold')
    plt.title(f"Eigenfrequency Spectrum for {shape.capitalize()} Shape", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(path, dpi=300)
    plt.show()

if __name__ == "__main__":
    # Example usage:
    L_values = [1, 2, 3, 4, 5]  # Different domain sizes
    N = 20  # Grid size
    shape = 'square'  # shape = 'rectangle' or 'circle'

    spectrum = compute_eigenfrequency_spectrum(L_values, N, "circle", num_eigenvalues=10, method="eigh")
    plot_eigenfrequency_spectrum_by_shape(spectrum, shape)