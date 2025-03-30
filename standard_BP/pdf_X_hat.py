
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

n=14
R=0.5
sigma=0.5
N=np.random.normal(0,1,n)
Wi=np.random.uniform(-10,10,n)
a=np.dot(Wi,N)
l2n=np.linalg.norm(Wi) #l2 norm
sigma_tilda=sigma*l2n

x=np.linspace(0.1,4,50)
h=[0,0.5,0.99]
l=R*n
t1=np.zeros(len(x))
t2=np.zeros(len(x))
t3=np.zeros(len(x))
fx=np.zeros((3,len(x)))

for j in range (3):
    for i in range (len(x)-1):
        num=(x[i]/l)/(1-h[j]*x[i]/l+h[j])
        t1[i]=(1+h[j])/(sigma_tilda*np.sqrt(2*math.pi))
        t2[i]=1/(x[i]/l*(1-h[j]*x[i]/l+h[j]))
        t3[i]=math.exp(-pow(np.log(num),2)/(2*pow(sigma_tilda,2)))
        fx[j,i]=t1[i]*t2[i]*t3[i]

fx[0,:]=fx[0,:]/max(fx[0,:])
fx[1,:]=fx[1,:]/max(fx[1,:])
fx[2,:]=fx[2,:]/max(fx[2,:])
#print('fx[0,:] ', fx[0,:])

fig1 = plt.figure(figsize=(8,6))
plt.plot(x[0:len(x)-2],fx[0,0:len(x)-2],'-.o',x[0:len(x)-2],fx[1,0:len(x)-2],'-.*',x[0:len(x)-2],fx[2,0:len(x)-2],'-.<',linewidth=3,markersize =10) 
#plt.semilogy(a1,del_sgd1_ag0,'-.o',a1,del_fm_ag0,'-.*',linewidth=3,markersize =10) 
#plt.ylim([1e-7, 1e-2])
#plt.xlim([1.74, 3.6])
plt.grid(True, which ="both")
#plt.title('BP using $(128,64)$ optimized polar code4',fontsize=16)
plt.ylabel('$f_{b_{\mathrm{min}}}(b_\mathrm{min})$',fontsize=20)
plt.xlabel('$b_{\mathrm{min}}$',fontsize=20)
plt.gca().legend(('$h=0$',
				  '$h=0.5$',
				  '$h=0.99$'),loc="upper right",labelspacing=0.25)
#plt.show()
fig1.savefig('pdf.pdf', bbox_inches='tight')


sigma=np.linspace(0.1,10,50)
sigma_tilda=sigma*l2n
I=np.zeros(len(sigma_tilda))
for i in range (len(sigma_tilda)):
    #I[i]=1/np.sqrt(2)*(math.erf(np.log(1/l)/sigma_tilda[i])+1)
    I[i]=1/2*(math.erf(np.log(1/l)/(np.sqrt(2)*sigma_tilda[i]))+1)
    
print('sigma: ',sigma)
print('I: ',I)

fig2 = plt.figure(figsize=(8,6))
plt.plot(sigma,I,'-.o',linewidth=3,markersize =10) 
#plt.semilogy(a1,del_sgd1_ag0,'-.o',a1,del_fm_ag0,'-.*',linewidth=3,markersize =10) 
#plt.ylim([1e-7, 1e-2])
#plt.xlim([1.74, 3.6])
plt.grid(True, which ="both")
#plt.title('BP using $(128,64)$ optimized polar code4',fontsize=16)
plt.ylabel(r'$\hat{F}_{b_\mathrm{min}}(1)$',fontsize=20)
plt.xlabel('$\sigma$',fontsize=20)
#plt.gca().legend(('$h=0$'),loc="upper right",labelspacing=0.25)
#plt.show()
fig2.savefig('prob.pdf', bbox_inches='tight')

