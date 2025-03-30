
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

a1=np.linspace(0, 1, num=25)
a2=np.linspace(-1, 0, num=25)

del_sgd1_ag0=2*np.log(1+np.exp(a1));
del_sgd1_al0=2*np.log(1+np.exp(-1*a2));

del_sgd1_ag0b=-del_sgd1_ag0/2;
del_sgd1_al0b=-del_sgd1_al0/2

del_fm_ag0=2*(np.exp(2*a1)+np.exp(a1)+1)/pow((1+np.exp(a1)),2)
del_fm_al0=2*(2*np.exp(a2)+1)/pow((1+np.exp(a2)),2)

del_fm_ag0b=-del_fm_ag0/2
del_fm_al0b=-del_fm_al0/2


fig1 = plt.figure(figsize=(8,6))
#plt.plot(a1,del_sgd1_ag0,'-.o',a1,del_sgd1_ag0b,'-.o',a1,del_fm_ag0,'-.*',a1,del_fm_ag0b,'-.*',linewidth=3,markersize =10) 
plt.plot(a1,del_sgd1_ag0,'-.o',a1,del_fm_ag0,'-.*',linewidth=3,markersize =10) 
#plt.semilogy(a1,del_sgd1_ag0,'-.o',a1,del_fm_ag0,'-.*',linewidth=3,markersize =10) 
#plt.ylim([1e-7, 1e-2])
#plt.xlim([1.74, 3.6])
plt.grid(True, which ="both")
#plt.title('BP using $(128,64)$ optimized polar code4',fontsize=16)
plt.ylabel('sensitivity',fontsize=20)
plt.xlabel('$a$',fontsize=20)
'''
plt.gca().legend(('$S_{\mathrm{SGD(1)}}^{(i)}$ UB',
				  '$S_{\mathrm{SGD(1)}}^{(i)}$ LB',
                  '$S_{\mathrm{FM}}^{(i)}$ UB',
                  '$S_{\mathrm{FM}}^{(i)}$ LB'),loc="upper right",labelspacing=0.25)
'''
plt.gca().legend(('$\Delta_{\mathrm{SGD(1)}}^{(i)}$',
				  '$\Delta_{\mathrm{FM}}^{(i)}$'),loc="upper left",labelspacing=0.25)
#plt.show()
fig1.savefig('fm_vs_sgd1_ag0.pdf', bbox_inches='tight')


fig2 = plt.figure(figsize=(8,6))
#plt.plot(a2,del_sgd1_al0,'-.o',a2,del_sgd1_al0b,'-.o',a2,del_fm_al0,'-.*',a2,del_fm_al0b,'-.*',linewidth=3,markersize =10) 
plt.plot(a2,del_sgd1_al0,'-.o',a2,del_fm_al0,'-.*',linewidth=3,markersize =10) 
#plt.ylim([1e-7, 1e-2])
#plt.xlim([1.74, 3.6])
plt.grid(True, which ="both")
plt.ylabel('sensitivity',fontsize=20)
plt.xlabel('$a$',fontsize=20)
plt.gca().legend(('$\Delta_{\mathrm{SGD(1)}}^{(i)}$',
				  '$\Delta_{\mathrm{FM}}^{(i)}$'),loc="upper right",labelspacing=0.25)
#plt.show()
fig2.savefig('fm_vs_sgd1_al0.pdf', bbox_inches='tight')
      

