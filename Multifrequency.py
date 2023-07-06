# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:05:43 2023

@author: Jason
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

B_r = np.array([1.2+1.6, 16.8, 170.8+45.8, 10.5, 0])
B_phi = np.array([1.2, 11.2, 85.4+19.1, 0, 0])
B_theta = np.array([0, 1.3+2.2, 16, 15.8+0.9,1.1])

T = np.array([3.74, 5.62, 11.23, 85.22, 641.90]) * 3600

omega_m = 2*np.pi/T

sigma_res = 30
sigma_oc = 0.5

r_0 = 1520e3
r_1 = 1410e3
r_m = 1560e3
r_res = 20e3
r_c = r_m-r_res
th_c = np.pi/2
lam_c = np.pi/2

mu_0 = 4*np.pi *1e-7

k_oc = (1+1j)*np.sqrt(mu_0 * sigma_oc * omega_m/2)
k_res = (1+1j)*np.sqrt(mu_0 * sigma_res * omega_m/2)

z_0 = r_0 * k_oc
z_1 = r_1 * k_oc
z_res = r_res * k_res
d = z_0 - z_1

l=1

R = z_1 * special.jv(-l-3/2, z_1)/((2*l + 1)*special.jv(l+1/2, z_1) - z_1 * special.jv(l-1/2, z_1)) 
    
BiBe_res = -l/(l+1) * special.jv(l+3/2, z_res)/special.jv(l-1/2, z_res)
                     
BiBe =  -l/(l+1)*(R * special.jv(l+3/2, z_0) - special.jv(-l-3/2, z_0))/(
 	R * special.jv(l-1/2, z_0) - special.jv(-l+1/2, z_0) )

A_oc = 2*np.abs(BiBe)
A_res = 2*np.abs(BiBe_res)

fig, ax = plt.subplots(3,1,figsize=(9,9))
ax[0].scatter(T/3600, B_r, color = 'k', marker = '^', s=40, label = r'$B_r$')
ax[0].scatter(T/3600, B_theta, color = 'violet', marker = '^', s=40, label = r'$B_\theta$')
ax[0].scatter(T/3600, B_phi, color = 'blue', marker = '^', s=40, label = r'$B_\lambda$')
ax[1].plot(T/3600, A_res, color = 'k')
ax[2].scatter(T/3600, A_res*B_r, color = 'k', marker = '^', s=40)
ax[2].scatter(T/3600, A_res*B_theta, color = 'violet', marker = '^', s=40)
ax[2].scatter(T/3600, A_res*B_phi, color = 'blue', marker = '^', s=40)
for i in range(3):
    ax[i].set_yscale('log')
    ax[i].set_xscale('log')
    ax[i].tick_params('both', labelsize = 13)
#ax[0].set_ylim(5e-4,5e0)
ax[0].set(xticklabels=[])
ax[1].set(xticklabels=[])
ax[2].set_xlabel(r'Period $T$ /h', fontsize = 14)
ax[0].set_ylabel(r'$B_i$ /nT', fontsize = 14)
ax[1].set_ylabel(r'$A_{res}$', fontsize = 14)
ax[2].set_ylabel(r'$A_{res} B_i$ /nT', fontsize = 14)
ax[0].legend(frameon=False,fontsize=15)
#plt.savefig('Multifrequency.pdf', bbox_inches='tight')