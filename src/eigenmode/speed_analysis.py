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

if __name__ == "__main__":
    N = 20  # Grid size
    num_repeats = 10

    print("Running benchmark...")
    times = benchmark(N, num_repeats=num_repeats)
    results = analyze_times(times)

    # print the results
    for method, stats in results.items():
        print(f"\nMethod: {method}")
        print(f"  Mean Time: {stats['mean']:.6f} sec")
        print(f"  Std Dev  : {stats['std_dev']:.6f} sec")
        print(f"  95% CI   : {stats['confidence_interval']}")

    # plot the results
    plot_results(results)