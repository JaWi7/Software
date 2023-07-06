# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:22:04 2023

@author: Me """

import numpy as np
import matplotlib.pyplot as plt


##############################SURFACE RESULTS##################################


Bdiff_0 =  np.array([[0.135, 1.35, 2.690, 4.018, 5.319, 7.853], #20km
                    [9.81e-2, 0.980, 1.954, 2.924, 3.885, 5.768], #17.5km
                    [6.8e-2, 0.680, 1.358, 2.032, 2.706, 4.036], #15km
                    [5.78e-2, 0.577, 1.152, 1.725, 2.296,  3.431], #14km
                    [4.85e-2, 0.484, 0.968, 1.450, 1.930, 2.888], #13km
                    [4.017e-2, 0.402, 0.804, 1.205, 1.605, 2.401], #12km
                    [3.29e-2, 0.329, 0.657, 0.985, 1.312, 1.963], #11km
                    [0.0264,0.264, 0.528, 0.792, 1.055, 1.580], #10m
                    [2.08e-2,0.207, 0.414, 0.622, 0.829, 1.242], #9km
                    [1.59e-2,0.159, 0.318, 0.477, 0.635, 0.953], #8km
                    [1.18e-2,0.118, 0.236, 0.354, 0.473, 0.709], #7km
                    [8.39e-3, 0.084, 0.168, 0.252, 0.336, 0.504], #6km
                    [5.66e-3, 5.6e-2, 0.113, 0.170, 0.226, 0.340]]) #5km


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
#plt.savefig('Bdiff_0.pdf', bbox_inches = 'tight')

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