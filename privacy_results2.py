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

fig1 = plt.figure(figsize=(8, 6))
x_indices = range(len(eps))

FM_noiseless = np.zeros(len(eps))
FM_noisy2 = np.zeros(len(eps))
FM_time1 = np.zeros(len(eps))
FM_time2 = np.zeros(len(eps))

for i in range(len(eps)):
    accuracy = np.loadtxt(f"results/FMaccuracy_noislessInp_{i}.txt")
    FM_noiseless[i] = np.array(accuracy).reshape(1,-1).mean(1)

    accuracy = np.loadtxt(f"results/FMaccuracy_noisyInp_{i}_1_5.txt")
    FM_noisy2[i] = np.array(accuracy).reshape(1,-1).mean(1)

    time_ = np.loadtxt(f"results/fm_time_{i}_1_0_1.txt")
    FM_time1[i] = np.array(time_).reshape(1,-1).mean(1)

    time_ = np.loadtxt(f"results/fm_time_{i}_1_1_5.txt")
    FM_time2[i] = np.array(time_).reshape(1,-1).mean(1)
    

'''
plt.plot(
    x_indices, FM_noiseless, '-.*',
    x_indices, FM_noisy2, '-.d',
    linewidth=3, markersize=10
)
'''

for u in users:
    folder = folders[u]
    c = colors[u]

    # Initialize arrays
    dpsgd2 = np.zeros(len(eps))
    dpsgd3 = np.zeros(len(eps))
    dpsgd_time2 = np.zeros(len(eps))
    dpsgd_time3 = np.zeros(len(eps))

    PALM_noiseless = np.zeros(len(eps))
    PALM_noisy2 = np.zeros(len(eps))
    PALM_time = np.zeros(len(eps))

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

    # --- Load PALM accuracy + time ---
    for i in range(len(eps)):
        accuracy = np.loadtxt(f"{folder}/PALMaccuracy_noislessInp_{i}.txt")
        PALM_noiseless[i] = np.array(accuracy).reshape(1,-1).mean(1)

        accuracy = np.loadtxt(f"{folder}/PALMaccuracy_noisyInp_{i}_1_5.txt")
        PALM_noisy2[i] = np.array(accuracy).reshape(1,-1).mean(1)

        time_ = np.loadtxt(f"{folder}/PALM_time_{i}_1_0_5.txt")
        PALM_time[i] = np.array(time_).reshape(1,-1).mean(1)

    # --- Plot results for this user count ---
    plt.plot(x_indices, dpsgd2, c+'-.s', linewidth=3, markersize=10)
    plt.plot(x_indices, dpsgd3, c+'-.x', linewidth=3, markersize=10)
    plt.plot(x_indices, PALM_noiseless, c+'-.^', linewidth=3, markersize=10)
    plt.plot(x_indices, PALM_noisy2, c+'-.h', linewidth=3, markersize=10)
    
plt.xticks(x_indices, eps)
plt.ylim([0.8, 0.97])
plt.gca().invert_xaxis()
plt.grid(True, which="both")
plt.ylabel('accuracy', fontsize=16)
plt.xlabel('privacy budget $\\epsilon$', fontsize=16)
plt.gca().legend((
    'FM noiseless inputs',
    'FM noisy inputs',
    'DP-SGD noiseless inputs ($m=2$)',
    'DP-SGD noiseless inputs ($m=5$)',
    'DP-SGD noiseless inputs ($m=10$)',
    'DP-SGD noiseless inputs ($m=20$)',
    'DP-SGD noisy inputs ($m=2$)',
    'DP-SGD noisy inputs ($m=5$)',
    'DP-SGD noisy inputs ($m=10$)',
    'DP-SGD noisy inputs ($m=20$)',
    'SPOF noiseless inputs ($m=2$)',
    'SPOF noiseless inputs ($m=5$)',
    'SPOF noiseless inputs ($m=10$)',
    'SPOF noiseless inputs ($m=20$)',
    'SPOF noisy inputs ($m=2$)',
    'SPOF noisy inputs ($m=5$)',
    'SPOF noisy inputs ($m=10$)',
    'SPOF noisy inputs ($m=20$)'), loc="center", bbox_to_anchor=(0.325, -0.1), borderaxespad=0., labelspacing=0.2)
fig1.savefig('results.pdf', bbox_inches='tight')



# Average ONLY times
FM_time_avg = (FM_time1 + FM_time2) / 2

DPSGD_time_avg = (dpsgd_time2 + dpsgd_time3) / 2
PALM_time_avg = PALM_time
NonPrivate_time_avg = nonprivate_time

# Build table
table = np.vstack([
    eps,
    FM_time_avg,
    DPSGD_time_avg,
    PALM_time_avg,
    NonPrivate_time_avg
])

df = pd.DataFrame(
    table,
    index=["epsilon", "FM", "DPSGD", "PALM", "NonPrivate"]
)

print(df.to_string())

