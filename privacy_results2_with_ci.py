
import os
os.system('clear')  

import numpy as np
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
plt.rcParams.update({'font.size': 16})
start_time = time.time()
from scipy.interpolate import make_interp_spline

eps=[0.001, 0.01, 0.1, 1.0, 10.0] #0.0018, 0.0032, 0.0056, 
FM_noisy=np.zeros(len(eps))
FM_noisy2=np.zeros(len(eps))
FM_noiseless=np.zeros(len(eps))
dpsgd=np.zeros(len(eps))
dpsgd2=np.zeros(len(eps))
dpsgd3=np.zeros(len(eps))
nonprivate=np.zeros(len(eps))

nonprivate=np.zeros(len(eps))

dpsgd_mean = np.zeros(len(eps))
dpsgd_std = np.zeros(len(eps))

for i in range(len(eps)):
    accuracy = np.loadtxt(f"results2/dpsgdaccuracy_{i}_0_0_1.txt")
    accuracy = np.array(accuracy).reshape(1, -1)
    dpsgd_mean[i] = accuracy.mean()
    dpsgd_std[i] = accuracy.std()

    
for i in range(len(eps)):
    accuracy=np.loadtxt("results/nonprivate_0.txt") 
    accuracy=array(accuracy).reshape(1, -1)
    nonprivate[i]=accuracy.mean(1)
    
'''
a=FM_noiseless.mean(0)
b=FM_noisy.mean(0)
c=FM_noisy2.mean(0)
print('a,b,c',a,b,c)
'''

n = 30  # number of Monte Carlo runs
dpsgd_conf = 1.96 * (dpsgd_std / np.sqrt(n))
    
fig1 = plt.figure(figsize=(8,6))
x_indices = range(len(eps))
# Plot using indices
#plt.plot(x_indices, nonprivate, '-.o', linewidth=3, markersize=10)
plt.plot(x_indices, nonprivate, '-.o', linewidth=3, markersize=10)
plt.errorbar(x_indices, dpsgd_mean, yerr=dpsgd_conf, fmt='-.*', capsize=5, linewidth=3, markersize=10, label='DP-SGD (with CI)')
# Set custom x-ticks with original x-values
plt.xticks(x_indices, eps)

'''
diff1=(dpsgd2.mean(0)-FM_noiseless.mean(0))*100
print('diff1: ',diff1)
diff2=(dpsgd3.mean(0)-FM_noisy2.mean(0))*100
print('diff2: ',diff2)
'''

plt.ylim([0.825, 0.86])
#plt.xlim([1.74, 3.6])
plt.gca().invert_xaxis()
#plt.gca().invert_yaxis()
plt.grid(True, which ="both")
#plt.title('Results Using BK and GC Based on FastDP for $m=2$',fontsize=16)
plt.ylabel('accuracy',fontsize=20)
plt.xlabel('privacy budget $\epsilon$',fontsize=20)
plt.gca().legend(('non-private (standard BP)',
                  'DP-SGD noiseless inputs (BK+GC)',
				  'DP-SGD noisy inputs, $\sigma=5$ (BK+GC)'),
				  loc="center", bbox_to_anchor=(0.5, 0.8), borderaxespad=0.,labelspacing=0.25
				  )
#plt.show()
fig1.savefig('results2.pdf', bbox_inches='tight')


