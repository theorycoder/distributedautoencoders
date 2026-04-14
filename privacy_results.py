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

#eps=[0.001, 0.01, 0.1, 1.0, 10.0] #use the results2 folder
eps=[0.1, 0.2, 0.4, 0.8, 1.6]
FM_noisy=np.zeros(len(eps))
FM_noisy2=np.zeros(len(eps))
FM_noiseless=np.zeros(len(eps))
dpsgd=np.zeros(len(eps))
dpsgd2=np.zeros(len(eps))
dpsgd3=np.zeros(len(eps))
nonprivate=np.zeros(len(eps))
nonprivate_time=np.zeros(len(eps))
FM_time=np.zeros(len(eps))
dpsgd_time=np.zeros(len(eps))

# --- PALM arrays ---
PALM_noisy=np.zeros(len(eps))
PALM_noisy2=np.zeros(len(eps))
PALM_noiseless=np.zeros(len(eps))
PALM_time=np.zeros(len(eps))

# --- Load FM accuracy ---
for i in range(len(eps)):
    accuracy=np.loadtxt(f"results_2/FMaccuracy_noislessInp_{i}.txt") 
    FM_noiseless[i]=array(accuracy).reshape(1, -1).mean(1)

    #accuracy=np.loadtxt(f"results_2/FMaccuracy_noisyInp_{i}_1_1.txt") 
    #FM_noisy[i]=array(accuracy).reshape(1, -1).mean(1)

    accuracy=np.loadtxt(f"results_2/FMaccuracy_noisyInp_{i}_1_5.txt") 
    FM_noisy2[i]=array(accuracy).reshape(1, -1).mean(1)

# --- Load DPSGD accuracy ---
for i in range(len(eps)):
    accuracy=np.loadtxt(f"results_2/dpsgdaccuracy_{i}_1_0_1.txt") 
    dpsgd2[i]=array(accuracy).reshape(1, -1).mean(1)

    accuracy=np.loadtxt(f"results_2/dpsgdaccuracy_{i}_1_1_5.txt") 
    dpsgd3[i]=array(accuracy).reshape(1, -1).mean(1) 

# --- Load Non-private ---
for i in range(len(eps)):
    accuracy=np.loadtxt("results_2/nonprivate_0.txt") 
    nonprivate[i]=array(accuracy).reshape(1, -1).mean(1)

# --- Load time: non-private ---
for i in range(1):
    time_=np.loadtxt(f"results_2/nonPrivate_time_{i}_1_1_5.txt") 
    nonprivate_time[i]=array(time_).reshape(1, -1).mean(1)

# --- Load time: FM ---
for i in range(len(eps)):
    time_=np.loadtxt(f"results_2/FM_time_{i}_1_0_5.txt") 
    FM_time[i]=array(time_).reshape(1, -1).mean(1)

# --- Load time: DPSGD ---
for i in range(len(eps)):
    time_=np.loadtxt(f"results_2/dpsgd_time_{i}_1_0_1.txt") 
    dpsgd_time[i]=array(time_).reshape(1, -1).mean(1)

# --- Load PALM results ---
for i in range(len(eps)):
    accuracy=np.loadtxt(f"results_2/PALMaccuracy_noislessInp_{i}.txt")
    PALM_noiseless[i]=array(accuracy).reshape(1, -1).mean(1)

    #accuracy=np.loadtxt(f"results_2/PALMaccuracy_noisyInp_{i}_1_1.txt")
    #PALM_noisy[i]=array(accuracy).reshape(1, -1).mean(1)

    accuracy=np.loadtxt(f"results_2/PALMaccuracy_noisyInp_{i}_1_5.txt")
    PALM_noisy2[i]=array(accuracy).reshape(1, -1).mean(1)

    time_=np.loadtxt(f"results_2/PALM_time_{i}_1_0_5.txt")
    PALM_time[i]=array(time_).reshape(1, -1).mean(1)

# --- Optional debug prints ---
diff1=(PALM_noiseless.mean(0)-dpsgd2.mean(0))/dpsgd2.mean(0)*100
print('diff1: ',diff1)
diff2=(PALM_noisy2.mean(0)-dpsgd3.mean(0))/dpsgd3.mean(0)*100
print('diff2: ',diff2)
#diff2=(dpsgd_time.mean(0)-PALM_time.mean(0))*100
print('dpsgd_time: ',dpsgd_time)
print('PALM_time: ',PALM_time)
time_diff1=(dpsgd_time.mean(0)-PALM_time.mean(0))/dpsgd_time.mean(0)*100
print('time_diff1: ',time_diff1)

# --- Plot ---
fig1 = plt.figure(figsize=(8,6))
x_indices = range(len(eps))

plt.plot(
    x_indices, nonprivate, '-.o',
    x_indices, FM_noiseless, '-.*',
    #x_indices, FM_noisy, '-.<',
    x_indices, FM_noisy2, '-.d',
    x_indices, dpsgd2, '-.s',
    x_indices, dpsgd3, '-.x',
    x_indices, PALM_noiseless, '--^',
    #x_indices, PALM_noisy, '--v',
    x_indices, PALM_noisy2, '--h',
    linewidth=3, markersize=10
)

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
    'SPOF noiseless inputs',
    #'PALM noisy inputs, $\sigma=1$',
    'SPOF noisy inputs, $\sigma=5$'
), loc="center", bbox_to_anchor=(0.325, 0.2), borderaxespad=0., labelspacing=0.25)

fig1.savefig('results.pdf', bbox_inches='tight')

# Average ONLY times
FM_time_avg = FM_time
DPSGD_time_avg = dpsgd_time
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
