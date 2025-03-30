
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

#snr22=np.loadtxt("22/snr22.txt") 
C=4
a=np.linspace(-2, 10, num=50)

del_fm_ag0=2*(np.exp(2*a)+np.exp(a)+1)/pow((1+np.exp(a)),2)
del_fm_al0=2*(2*np.exp(a)+1)/pow((1+np.exp(a)),2)

del_fm_ag0b=-del_fm_ag0/2
del_fm_al0b=-del_fm_al0/2

A=(np.log(1+np.exp(-a)))/(np.log(1+np.exp(a)))
y2=(1+np.exp(-a))/(1+np.exp(a))
y2b=(1+np.exp(-a))/(1+np.exp(a))

delta_fm_i=np.log(1+np.exp(a))+2*(A +(2*np.exp(a))/pow((1+np.exp(a)),2) +(np.exp(a))/pow((1+np.exp(a)),1))

delta_sgd_i_1=(2*+np.exp(a))/(1+np.exp(a))
delta_sgd_i_2=2*C*np.ones(len(delta_sgd_i_1))

fig1 = plt.figure(figsize=(8,6))
#plt.plot(a,y2,'-.o',a,y2b,'-.*',linewidth=3,markersize =10) 
plt.plot(a,delta_sgd_i_1,'-.>',a,delta_sgd_i_2,'-.o',a,delta_fm_i,'-.*',linewidth=3,markersize =10)
#plt.ylim([1e-7, 1e-2])
#plt.xlim([1.74, 3.6])
plt.grid(True, which ="both")
#plt.title('BP using $(128,64)$ optimized polar code4',fontsize=16)
plt.ylabel('sensitivity upper bound',fontsize=20)
plt.xlabel('$a$',fontsize=20)

plt.gca().legend(('$\Delta_{\mathrm{SGD(1)}}(i)$',
                  '$\Delta_{\mathrm{SGD(2)}}(i)$',
				  '$\Delta_{\mathrm{FM}}(i)$'),loc="upper right",labelspacing=0.25)

#plt.show()
fig1.savefig('sens_ub.pdf', bbox_inches='tight')

