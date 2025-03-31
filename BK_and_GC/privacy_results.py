
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

eps=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
FM_noisy=np.zeros(len(eps))
FM_noisy2=np.zeros(len(eps))
FM_noiseless=np.zeros(len(eps))
dpsgd=np.zeros(len(eps))
dpsgd2=np.zeros(len(eps))
dpsgd3=np.zeros(len(eps))
nonprivate=np.zeros(len(eps))

nonprivate=np.zeros(len(eps))

for i in range(len(eps)):
    if i==0:
        accuracy=np.loadtxt("results/FMaccuracy_noislessInp_0.txt") #results2 folder has results with 4 users (autoencoder4)
    elif i==1:
        accuracy=np.loadtxt("results/FMaccuracy_noislessInp_1.txt") 
    elif i==2:
        accuracy=np.loadtxt("results/FMaccuracy_noislessInp_2.txt") 
    elif i==3:
        accuracy=np.loadtxt("results/FMaccuracy_noislessInp_3.txt") 
    elif i==4:
        accuracy=np.loadtxt("results/FMaccuracy_noislessInp_4.txt") 
    elif i==5:
        accuracy=np.loadtxt("results/FMaccuracy_noislessInp_5.txt") 
    elif i==6:
        accuracy=np.loadtxt("results/FMaccuracy_noislessInp_6.txt") 
    accuracy=array(accuracy).reshape((1,14))
    FM_noiseless[i]=accuracy.mean(1)

    #using delta_tildaFM, input noise sigma=1
    if i==0:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_0_1_1.txt") 
    elif i==1:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_1_1_1.txt") 
    elif i==2:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_2_1_1.txt") 
    elif i==3:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_3_1_1.txt") 
    elif i==4:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_4_1_1.txt") 
    elif i==5:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_5_1_1.txt") 
    elif i==6:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_6_1_1.txt") 
    accuracy=array(accuracy).reshape((1,14))
    FM_noisy[i]=accuracy.mean(1)
    
    #using delta_tildaFM, input noise sigma=5
    if i==0:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_0_1_5.txt") 
    elif i==1:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_1_1_5.txt") 
    elif i==2:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_2_1_5.txt") 
    elif i==3:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_3_1_5.txt") 
    elif i==4:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_4_1_5.txt") 
    elif i==5:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_5_1_5.txt") 
    elif i==6:
        accuracy=np.loadtxt("results/FMaccuracy_noisyInp_6_1_5.txt") 
    accuracy=array(accuracy).reshape((1,14))
    FM_noisy2[i]=accuracy.mean(1)
    
    
    #dpsgd using built-in loss, C=4, delta_sgd=2nlog2,
    if i==0:
        accuracy=np.loadtxt("results/dpsgdaccuracy_0_0_0_5.txt") #noiseless
    elif i==1:
        accuracy=np.loadtxt("results/dpsgdaccuracy_1_0_0_5.txt") 
    elif i==2:
        accuracy=np.loadtxt("results/dpsgdaccuracy_2_0_0_5.txt") 
    elif i==3:
        accuracy=np.loadtxt("results/dpsgdaccuracy_3_0_0_5.txt") 
    elif i==4:
        accuracy=np.loadtxt("results/dpsgdaccuracy_4_0_0_5.txt") 
    elif i==5:
        accuracy=np.loadtxt("results/dpsgdaccuracy_5_0_0_5.txt") 
    elif i==6:
        accuracy=np.loadtxt("results/dpsgdaccuracy_6_0_0_5.txt") 
    accuracy=array(accuracy).reshape((1,14))
    dpsgd2[i]=accuracy.mean(1)
    
    #dpsgd using built-in loss, C=4, delta_sgd=2nlog2 without sensor noise
    if i==0:
        accuracy=np.loadtxt("results/dpsgdaccuracy_0_0_1_5.txt") 
    elif i==1:
        accuracy=np.loadtxt("results/dpsgdaccuracy_1_0_1_5.txt") 
    elif i==2:
        accuracy=np.loadtxt("results/dpsgdaccuracy_2_0_1_5.txt") 
    elif i==3:
        accuracy=np.loadtxt("results/dpsgdaccuracy_3_0_1_5.txt") 
    elif i==4:
        accuracy=np.loadtxt("results/dpsgdaccuracy_4_0_1_5.txt") 
    elif i==5:
        accuracy=np.loadtxt("results/dpsgdaccuracy_5_0_1_5.txt") 
    elif i==6:
        accuracy=np.loadtxt("results/dpsgdaccuracy_6_0_1_5.txt") 
    accuracy=array(accuracy).reshape((1,14))
    dpsgd3[i]=accuracy.mean(1) 
    
for i in range(len(eps)):
    accuracy=np.loadtxt("results/accuracy_nonprivate0.txt") 
    accuracy=array(accuracy).reshape((1,7))
    nonprivate[i]=accuracy.mean(1)
    

a=FM_noiseless.mean(0)
b=FM_noisy.mean(0)
c=FM_noisy2.mean(0)
print('a,b,c',a,b,c)


    
fig1 = plt.figure(figsize=(8,6))
x_indices = range(len(eps))
# Plot using indices
plt.plot(x_indices,nonprivate,'-.o',x_indices,FM_noiseless,'-.*',x_indices,FM_noisy,'-.<',x_indices,FM_noisy2,'-.s',
         x_indices,dpsgd2,'-.x',x_indices,dpsgd3,'-.x',linewidth=3,markersize =10) 
# Set custom x-ticks with original x-values
plt.xticks(x_indices, eps)

diff1=(dpsgd2.mean(0)-FM_noiseless.mean(0))*100
print('diff1: ',diff1)
diff2=(dpsgd3.mean(0)-FM_noisy2.mean(0))*100
print('diff2: ',diff2)

#plt.ylim([1e-7, 1e-2])
#plt.xlim([1.74, 3.6])
plt.gca().invert_xaxis()
#plt.gca().invert_yaxis()
plt.grid(True, which ="both")
plt.title('Results Using BK and GC Based on FastDP for $m=2$',fontsize=16)
plt.ylabel('accuracy',fontsize=20)
plt.xlabel('privacy budget $\epsilon$',fontsize=20)
plt.gca().legend(('non-private (standard BP)',
				  'FM noiseless inputs (BK)',
				  'FM noisy inputs, $\sigma=1$ (BK)',
				  'FM noisy inputs, $\sigma=5$ (BK)',
				  'DP-SGD noiseless inputs (BK+GC)',
				  'DP-SGD noisy inputs, $\sigma=5$ (BK+GC)'),loc="center", bbox_to_anchor=(0.5, 0.6), borderaxespad=0.,labelspacing=0.25)
#plt.show()
fig1.savefig('results.pdf', bbox_inches='tight')


