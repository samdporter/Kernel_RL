#!/usr/bin/env python3
"""
Find the lowest NRMSE value across all spheres results.
"""
import pandas as pd
from pathlib import Path
import sys

def find_lowest_nrmse(results_dir: str = "results"):
    """Find the lowest NRMSE value across all spheres results."""
    results_path = Path(results_dir)

    # Find all nrmse.csv files in spheres directories
    nrmse_files = list(results_path.glob("spheres*/*nrmse.csv"))

    if not nrmse_files:
        print(f"No NRMSE files found in {results_dir}/spheres*/")
        return

    print(f"Found {len(nrmse_files)} NRMSE files\n")

    # Track global minimum
    global_min_nrmse = float('inf')
    global_min_info = None

    # Process each file
    results = []
    for csv_file in sorted(nrmse_files):
        try:
            df = pd.read_csv(csv_file)

            # Find minimum NRMSE in this file
            min_nrmse = df['nrmse'].min()
            min_iter = df.loc[df['nrmse'].idxmin(), 'iteration']

            result = {
                'experiment': csv_file.parent.name,
                'file': csv_file.name,
                'min_nrmse': min_nrmse,
                'iteration': int(min_iter),
                'path': str(csv_file)
            }
            results.append(result)

            # Update global minimum
            if min_nrmse < global_min_nrmse:
                global_min_nrmse = min_nrmse
                global_min_info = result

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    # Sort results by NRMSE
    results.sort(key=lambda x: x['min_nrmse'])

    # Print all results
    print("=" * 100)
    print(f"{'Experiment':<60} {'Min NRMSE':<12} {'Iteration':<10}")
    print("=" * 100)

    for r in results:
        print(f"{r['experiment']:<60} {r['min_nrmse']:<12.8f} {r['iteration']:<10}")

    print("=" * 100)
    print("\n🏆 LOWEST NRMSE:")
    print(f"  Value: {global_min_info['min_nrmse']:.8f}")
    print(f"  Iteration: {global_min_info['iteration']}")
    print(f"  Experiment: {global_min_info['experiment']}")
    print(f"  File: {global_min_info['path']}")
    print("=" * 100)

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    find_lowest_nrmse(results_dir)
