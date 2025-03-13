import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import src.eigenmode.gen_M as gen_M
import src.eigenmode.solve_eigenv as solve_eigenv

# time calculation for different methods
def benchmark(N, num_repeats=10, num_eigenvalues=6):
    M = gen_M.generate_M_with_square(N)

    methods = ["eig", "eigh", "eigs"]
    times = {method: [] for method in methods}

    for method in methods:
        for _ in range(num_repeats):
            start = time.perf_counter()
            solve_eigenv.solve_eigenproblem(M, method=method, num_eigenvalues=num_eigenvalues)
            end = time.perf_counter()
            times[method].append(end - start)

    return times

# statistical analysis for different methods
def analyze_times(times, confidence=0.95):
    results = {}
    for method, values in times.items():
        mean_time = np.mean(values)
        std_dev = np.std(values, ddof=1)
        n = len(values)
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin_of_error = t_critical * (std_dev / np.sqrt(n))
        confidence_interval = (mean_time - margin_of_error, mean_time + margin_of_error)
        
        results[method] = {
            "mean": mean_time,
            "std_dev": std_dev,
            "confidence_interval": confidence_interval
        }
    return results

def benchmark_set(N_list, num_repeats=5, num_eigenvalues=6):
    methods = ["eigh", "eigs"]
    times = {method: [] for method in methods}

    for N in N_list:
        M = gen_M.generate_M_with_square(N)

        for method in methods:
            time_measurements = []
            for _ in range(num_repeats):
                start = time.perf_counter()
                solve_eigenv.solve_eigenproblem(M, method=method, num_eigenvalues=num_eigenvalues)
                end = time.perf_counter()
                time_measurements.append(end - start)

            # Calculate mean, std dev, and confidence interval
            mean_time = np.mean(time_measurements)
            std_dev = np.std(time_measurements, ddof=1)
            n = len(time_measurements)
            t_critical = stats.t.ppf(0.975, df=n-1)  # 95% CI
            margin_of_error = t_critical * (std_dev / np.sqrt(n))
            confidence_interval = (mean_time - margin_of_error, mean_time + margin_of_error)

            times[method].append({
                "N": N,
                "mean": mean_time,
                "std_dev": std_dev,
                "confidence_interval": confidence_interval
            })
    
    return times

# plot the results
def plot_results(results):
    methods = list(results.keys())
    means = [results[m]["mean"] for m in methods]
    errors = [results[m]["confidence_interval"][1] - results[m]["mean"] for m in methods]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, means, yerr=errors, capsize=5, color=["blue", "green", "red"])
    plt.xlabel("Method")
    plt.ylabel("Time (seconds)")
    plt.title("Comparison of Eigenvalue Computation Methods")
    plt.show()

def plot_results_set(times, path="../../fig/eigenvalue_computation_time_error_bar.png"):
    plt.figure(figsize=(10, 6))

    markers = {"eig": "o-", "eigh": "s-", "eigs": "^-"}
    colors = {"eig": "red", "eigh": "green", "eigs": "blue"}

    for method in times.keys():
        N_values = [entry["N"] for entry in times[method]]
        means = [entry["mean"] for entry in times[method]]
        errors = [entry["confidence_interval"][1] - entry["mean"] for entry in times[method]]

        plt.errorbar(N_values, means, yerr=errors, fmt=markers[method], capsize=5, color=colors[method], label=f"{method} method time consumed (with 95% CI)")

        # label the confidence intervals
        for i, N in enumerate(N_values):
            if i % 3 == 0:
                ci_low, ci_high = times[method][i]["confidence_interval"]
                plt.annotate(f"[{ci_low:.3e}, {ci_high:.3e}]", (N, means[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color=colors[method])

    plt.xlabel("Grid Size (N)")
    plt.ylabel("Time (seconds) Consumed with 95% CI")
    plt.title("Eigenvalue Computation Time Error Bar for Different N")
    plt.yscale("log")  # Log scale for y-axis
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.show()

if __name__ == "__main__":
    N_list = np.linspace(10, 55, 10, dtype=int)  # 10 grid sizes from 10 to 55
    num_repeats = 5  # Number of repeats for each grid size

    print("Running benchmark...")
    times = benchmark_set(N_list, num_repeats=num_repeats)

    # Plot the results
    plot_results_set(times)