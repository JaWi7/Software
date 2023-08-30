# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:22:04 2023

Here, the results of our parameter study presented in the paper (Figures 8 and 11)
are listed.

"""

import numpy as np
import matplotlib.pyplot as plt


##############################SURFACE RESULTS##################################


Bdiff_0 =  np.array([[0.1647, 1.6448, 3.2794, 4.8936, 6.4891, 9.5747], #20km
                    [0.1157, 1.1554, 2.3064, 3.4510, 4.5822, 6.8079], #17.5km
                    [0.0772, 0.7714, 1.5409, 2.3062, 3.0693, 4.5798], #15km
                    [0.0644, 0.6435, 1.2861, 1.9264, 2.5631,  3.8298], #14km
                    [0.0532, 0.5315, 1.0627, 1.5928, 2.1209, 3.1681], #13km
                    [0.0434, 0.4344, 0.8684, 1.3021, 1.7348, 2.5955], #12km
                    [0.0350, 0.3496, 0.6986, 1.0472, 1.3958, 2.0908], #11km
                    [0.0277, 0.2772, 0.5541, 0.8306, 1.1066, 1.6585], #10m
                    [0.0215, 0.2152, 0.4303, 0.645, 0.8599, 1.2882], #9km
                    [0.0163, 0.1631, 0.3261, 0.4891, 0.6520, 0.9773], #8km
                    [0.0120, 0.1202, 0.2405, 0.3607, 0.4808, 0.7210], #7km
                    [0.0085, 0.0850, 0.1700, 0.2550, 0.3400, 0.5099], #6km
                    [0.0057, 0.0569, 0.1138, 0.1708, 0.2277, 0.3415]]) #5km

sigma = np.array([0.5,5,10,15,20,30])
r = np.array([20, 17.5, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5])
fig, ax = plt.subplots(figsize=(8,8))

for i in range(len(Bdiff_0[0])):
    if i == 0:
        ax.loglog(r, Bdiff_0[:,i], label = r'{} S/m'.format(sigma[i]))
    else:
        ax.loglog(r, Bdiff_0[:,i], label = r'{} S/m'.format(int(sigma[i])))
plt.xlim(5,20)
plt.hlines(1e0, 5, 20, color = 'k', linestyle = 'dashed')
plt.hlines(2.3e0, 5, 20, color = 'k', linestyle = 'dashed')
plt.legend(frameon=False, loc = 4, fontsize = 14)
plt.tick_params(axis='both', labelsize = 14)
plt.xlabel(r'$r_{res}$ /km', fontsize=15)
plt.ylabel(r'$\Delta B_r$ /nT', fontsize = 15)
plt.text(5.2,1.1, 'Ocean',fontsize=14)
plt.text(5.5,2.5, 'Small-scale random fluctuations',fontsize=14)
plt.savefig('Bdiff_0.pdf', bbox_inches = 'tight')

##############################25 km altitude###################################

deltaB_max = np.array([[3.59379342e-05, 3.59473456e-04, 7.19148296e-04, 1.07901226e-03, 1.43905307e-03, 2.15961608e-03],
                      [8.13983838e-05, 8.14288714e-04, 1.62921847e-03, 2.44473165e-03, 3.26077053e-03, 4.89419430e-03],
                      [0.0001606,  0.00160677,  0.00321522,0.00483, 0.00644, 0.00970],
                      [0.00029, 0.00289, 0.00579, 0.00870, 0.01163, 0.01751],
                      [0.00048, 0.00482,  0.00967, 0.01455, 0.01945, 0.02933],
                      [0.00076, 0.00760, 0.01525, 0.02295, 0.03071, 0.04634], #10
                      [0.00115, 0.01154, 0.02320, 0.03495, 0.04679, 0.07069], #11
                      [0.00164, 0.01651, 0.03317, 0.04997, 0.06688,  0.10098], #12
                      [0.00230, 0.02308, 0.04641, 0.06996, 0.09369, 0.14154], #13
                      [0.00312, 0.03140, 0.06320, 0.09533, 0.12774, 0.19307], #14
                      [0.00414, 0.04174, 0.08410, 0.12695, 0.17018, 0.25727], #15
                      [0.00770, 0.07781, 0.15715, 0.23761, 0.31878, 0.48154], #17.5
                      [0.01302, 0.13202, 0.26734, 0.40477, 0.54307, 0.81760]]) #20


r_ = np.array([5,6,7,8,9,10,11,12,13,14,15,17.5,20])
fig, ax = plt.subplots(figsize=(8,8))
for i in range(len(deltaB_max[0])):
    if i == 0:
        ax.loglog(r_, deltaB_max[:,i], label = r'{} S/m'.format(sigma[i]))
    else:
        ax.loglog(r_, deltaB_max[:,i], label = r'{} S/m'.format(int(sigma[i])))
plt.xlim(5,20)
plt.hlines(5e0, 5, 20, color = 'k', linestyle = 'dashed')
plt.legend(frameon=False, loc = 4, fontsize = 14)
plt.tick_params(axis='both', labelsize = 14)
plt.xlabel(r'$r_{res}$ /km', fontsize=15)
plt.ylabel(r'$\Delta B_r$ /nT', fontsize = 15)
plt.text(5.5,3.5, 'Small-scale random fluctuations',fontsize=14)
#plt.savefig('Bdiff_25.pdf',bbox_inches='tight')