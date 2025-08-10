import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

EPS_INDEX = "4"  # ε = 10
NUM_RUNS = 5    # Number of Monte Carlo runs per constant
NUM_WORKERS = 4 # Number of parallel workers

#constants = np.linspace(0.0, 2.5, 5)
constants = np.linspace(0.0, 5, 5)

def run_one_trial(c, run_id):
    env = os.environ.copy()
    env["DECODER_CONST"] = str(c)
    env["EPS_INDEX"] = EPS_INDEX
    env["MC_RUN"] = str(run_id)

    result = subprocess.run(
        ["python3", "autoencoder2_pytorch.py"],
        env=env,
        capture_output=True,
        text=True
    )

    # Parse accuracy from output
    combined_output = result.stdout + "\n" + result.stderr
    lines = combined_output.splitlines()
    acc_line = next((line for line in lines if "Accuracy" in line), None)

    if acc_line:
        try:
            acc = float(acc_line.split(":")[1].strip())
            return acc
        except:
            return float("nan")
    else:
        return float("nan")

# Run serially over constants
results = []
for c in constants:
    acc_vals = []
    print(f"Running simulations for constant: {c:.2f}")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_one_trial, c, i): i for i in range(NUM_RUNS)}
        for future in as_completed(futures):
            acc = future.result()
            acc_vals.append(acc)

    avg_acc = np.nanmean(acc_vals)
    results.append((c, avg_acc))
    print(f"Constant {c:.2f} -> Avg Accuracy: {avg_acc:.6f}")

# Sort results for plotting
results.sort()
x_vals, y_vals = zip(*results)

# Plot the results
fig = plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, marker='o')
plt.xlabel("Constant added to decoder weight")
plt.ylabel("Average Accuracy (1 - MSE)")
plt.title("FM Accuracy vs. Decoder Weight Constant (ε = 0.001)")
plt.grid(True)
plt.tight_layout()
fig.savefig('sweep_constant_full.pdf', bbox_inches='tight')

# Save results to file
with open("accuracy_vs_const_full.txt", "w") as f:
    f.write("Constant\tAverage_Accuracy\n")
    for c, acc in results:
        f.write(f"{c:.4f}\t{acc:.6f}\n")

