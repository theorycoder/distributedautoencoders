import os
os.system('clear')  

import matplotlib
matplotlib.use("Agg")  

import numpy as np
import pandas as pd
import time
import math
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})
plt.rcParams.update({'font.size': 14})
start_time = time.time()
from scipy.interpolate import make_interp_spline

# --- Settings ---
eps = [0.1, 0.2, 0.4, 0.8, 1.6]
users = [2, 5, 10, 20]
folders = {2: 'results_2', 5: 'results_5', 10: 'results_10', 20: 'results_20'}
colors = {2: 'k', 5: 'b', 10: 'g', 20: 'r'}

import numpy as np
import matplotlib.pyplot as plt

x_indices = range(len(eps))

# --- FM baseline results ---
FM_noiseless = np.zeros(len(eps))
FM_noisy2 = np.zeros(len(eps))
FM_time1 = np.zeros(len(eps))
FM_time2 = np.zeros(len(eps))

#these FM results are for one user only
for i in range(len(eps)):
    accuracy = np.loadtxt(f"results_2/FMaccuracy_noislessInp_{i}.txt")
    FM_noiseless[i] = np.array(accuracy).reshape(1,-1).mean(1)

    accuracy = np.loadtxt(f"results_2/FMaccuracy_noisyInp_{i}_1_5.txt")
    FM_noisy2[i] = np.array(accuracy).reshape(1,-1).mean(1)

    time_ = np.loadtxt(f"results_2/FM_time_{i}_1_0_5.txt")
    FM_time1[i] = np.array(time_).reshape(1,-1).mean(1)

    time_ = np.loadtxt(f"results_2/FM_time_{i}_1_1_5.txt")
    FM_time2[i] = np.array(time_).reshape(1,-1).mean(1)


# --- Same colors for every plot ---
curve_colors = ['k','b','r','g']   # black, blue, red, green


for u in users:

    fig = plt.figure(figsize=(8,6))

    folder = folders[u]

    # --- Initialize arrays ---
    dpsgd2 = np.zeros(len(eps))
    dpsgd3 = np.zeros(len(eps))
    dpsgd_time2 = np.zeros(len(eps))
    dpsgd_time3 = np.zeros(len(eps))

    DFM_noiseless = np.zeros(len(eps))
    DFM_noisy2 = np.zeros(len(eps))
    DFM_time = np.zeros(len(eps))

    nonprivate = np.zeros(len(eps))
    nonprivate_time = np.zeros(len(eps))

    # --- Load DPSGD accuracy + time ---
    for i in range(len(eps)):

        accuracy = np.loadtxt(f"{folder}/dpsgdaccuracy_{i}_1_0_1.txt")
        dpsgd2[i] = np.array(accuracy).reshape(1,-1).mean(1)

        accuracy = np.loadtxt(f"{folder}/dpsgdaccuracy_{i}_1_1_5.txt")
        dpsgd3[i] = np.array(accuracy).reshape(1,-1).mean(1)

        time_ = np.loadtxt(f"{folder}/dpsgd_time_{i}_1_0_1.txt")
        dpsgd_time2[i] = np.array(time_).reshape(1,-1).mean(1)

        time_ = np.loadtxt(f"{folder}/dpsgd_time_{i}_1_1_5.txt")
        dpsgd_time3[i] = np.array(time_).reshape(1,-1).mean(1)


    # --- Load Non-private accuracy + time ---
    for i in range(len(eps)):

        accuracy = np.loadtxt(f"{folder}/nonprivate_0.txt")
        nonprivate[i] = np.array(accuracy).reshape(1,-1).mean(1)

        time_ = np.loadtxt(f"{folder}/nonPrivate_time_0_1_1_5.txt")
        nonprivate_time[i] = np.array(time_).reshape(1,-1).mean(1)


    # --- Load DFM accuracy + time ---
    for i in range(len(eps)):

        accuracy = np.loadtxt(f"{folder}/DFMaccuracy_noislessInp_{i}.txt")
        DFM_noiseless[i] = np.array(accuracy).reshape(1,-1).mean(1)

        accuracy = np.loadtxt(f"{folder}/DFMaccuracy_noisyInp_{i}_1_5.txt")
        DFM_noisy2[i] = np.array(accuracy).reshape(1,-1).mean(1)

        time_ = np.loadtxt(f"{folder}/DFM_time_{i}_1_0_5.txt")
        DFM_time[i] = np.array(time_).reshape(1,-1).mean(1)


    # --- Plot curves (different color per curve) ---
    plt.plot(
    x_indices, nonprivate, '-.o',
    x_indices, FM_noiseless, '-.*',
    x_indices, FM_noisy2, '-.d',
    x_indices, dpsgd2, '-.s',
    x_indices, dpsgd3, '-.x',
    x_indices, DFM_noiseless, '--^',
    x_indices, DFM_noisy2, '--h',
    linewidth=3, markersize=10)


    plt.xticks(x_indices, eps)
    plt.ylim([0.7, 1])
    plt.gca().invert_xaxis()
    plt.grid(True, which="both")
    plt.ylabel('accuracy', fontsize=16)
    plt.xlabel('privacy budget $\epsilon$', fontsize=16)
    plt.gca().legend((
    'non-private',
    'FM noiseless inputs',
    #'FM noisy inputs, $\sigma=1$',
    'FM noisy inputs, $\sigma=5$',
    'DP-SGD noiseless inputs',
    'DP-SGD noisy inputs, $\sigma=5$',
    'DFM noiseless inputs',
    #'DFM noisy inputs, $\sigma=1$',
    'DFM noisy inputs, $\sigma=5$'), loc="center", bbox_to_anchor=(0.325, 0.2), borderaxespad=0., labelspacing=0.25)


    # --- Save figure ---
    plt.savefig(f"results_{u}.pdf", bbox_inches='tight')

    plt.close()



# Average ONLY times
FM_time_avg = (FM_time1 + FM_time2) / 2

DPSGD_time_avg = (dpsgd_time2 + dpsgd_time3) / 2
DFM_time_avg = DFM_time
NonPrivate_time_avg = nonprivate_time

# Build table
table = np.vstack([
    eps,
    FM_time_avg,
    DPSGD_time_avg,
    DFM_time_avg,
    NonPrivate_time_avg
])

df = pd.DataFrame(
    table,
    index=["epsilon", "FM", "DPSGD", "DFM", "NonPrivate"]
)

print(df.to_string())


# --- Collect all values across users ---
DFM_all = []
DPSGD_all = []
DFM_noisy_all = []
DFM_noiseless_all = []

for u in users:
    folder = folders[u]
    
    for i in range(len(eps)):
        # DFM
        dfm_nl = np.loadtxt(f"{folder}/DFMaccuracy_noislessInp_{i}.txt").mean()
        dfm_ny = np.loadtxt(f"{folder}/DFMaccuracy_noisyInp_{i}_1_5.txt").mean()
        
        # DPSGD
        dpsgd_nl = np.loadtxt(f"{folder}/dpsgdaccuracy_{i}_1_0_1.txt").mean()
        dpsgd_ny = np.loadtxt(f"{folder}/dpsgdaccuracy_{i}_1_1_5.txt").mean()
        
        # Store
        DFM_all.extend([dfm_nl, dfm_ny])
        DPSGD_all.extend([dpsgd_nl, dpsgd_ny])
        
        DFM_noiseless_all.append(dfm_nl)
        DFM_noisy_all.append(dfm_ny)

# Convert to numpy
DFM_all = np.array(DFM_all)
DPSGD_all = np.array(DPSGD_all)
DFM_noiseless_all = np.array(DFM_noiseless_all)
DFM_noisy_all = np.array(DFM_noisy_all)

# --- Compute averages ---
DFM_avg = DFM_all.mean()
DPSGD_avg = DPSGD_all.mean()

DFM_vs_DPSGD_gain = (DFM_avg - DPSGD_avg) / DPSGD_avg * 100

print(f"Average DFM accuracy: {DFM_avg:.4f}")
print(f"Average DPSGD accuracy: {DPSGD_avg:.4f}")
print(f"DFM is higher than DPSGD by: {DFM_vs_DPSGD_gain:.2f}%")




# Compute averages
DFM_noisy_avg = DFM_noisy_all.mean()

# You need to collect DPSGD noisy values (same way as DFM_noisy_all)
DPSGD_noisy_all = []

for u in users:
    folder = folders[u]
    for i in range(len(eps)):
        dpsgd_ny = np.loadtxt(f"{folder}/dpsgdaccuracy_{i}_1_1_5.txt").mean()
        DPSGD_noisy_all.append(dpsgd_ny)

DPSGD_noisy_all = np.array(DPSGD_noisy_all)
DPSGD_noisy_avg = DPSGD_noisy_all.mean()

# Compute percentage gain
DFM_vs_DPSGD_noisy_gain = (DFM_noisy_avg - DPSGD_noisy_avg) / DPSGD_noisy_avg * 100

print(f"Average DFM noisy accuracy: {DFM_noisy_avg:.4f}")
print(f"Average DPSGD noisy accuracy: {DPSGD_noisy_avg:.4f}")
print(f"DFM noisy vs DPSGD noisy gain: {DFM_vs_DPSGD_noisy_gain:.2f}%")
