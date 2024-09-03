# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:05:39 2022

This modul is used to provide geometrical and induction paraameters, as well as
the inducing background field.

"""

import numpy as np
import cmath
from scipy import special

###MAX DEGREE###

nlm = 2

###GRID PARAMETERS###

NTH = 180
NLAM = 360
B_TH = np.pi
B_PH = 2*np.pi
dth = B_TH/NTH
dlam = B_PH/NLAM


###GEOMETRICAL PARAMETERS###

r_0 = 1520e3 
r_1 = 1410e3
r_m = 1560e3
r_res = 20e3
r_c = r_m-r_res #distance between oc and res
th_c = np.pi/2
lam_c = 0# 0->along x-axis


###INDUCTION PARAMETERS###

T = 11.23 * 3600 #Synodic rotation period in s
omega_m = 2*np.pi/T
mu_0 = 4*np.pi *1e-7
sigma_oc = 0.455 # This value is adjusted manually so that A_oc = 0.91

"Depending on which script is used, provide integer for sigma_res or full array"

sigma_res = 30
#sigma_res = np.array([0.5,5,10,15,20,30])



k_oc = (1+1j)*np.sqrt(mu_0 * sigma_oc * omega_m/2)
k_res = (1+1j)*np.sqrt(mu_0 * sigma_res * omega_m/2)

z_0 = r_0 * k_oc
z_1 = r_1 * k_oc
z_res = r_res * k_res
d = z_0 - z_1


#Bi/Be for a homogeneous sphere 
BiBe_res = np.array([-l/(l+1) * special.jv(l+3/2, z_res)/special.jv(l-1/2, z_res)
                     for l in range(nlm+1)])

#Bi/Be and R (xi in Saur et al., 2010) for a spherical layer  
R = np.array([(z_1 * special.jv(-l-3/2, z_1))/((2*l + 1)*special.jv(l+1/2, z_1) - z_1 * special.jv(l-1/2, z_1)) 
              for l in range(nlm+1)])

BiBe =  np.array([-l/(l+1)*(R[l] * special.jv(l+3/2, z_0) - special.jv(-l-3/2, z_0))/(
 	R[l] * special.jv(l-1/2, z_0) - special.jv(-l+1/2, z_0) ) for l in range(nlm+1)])

# Analyical approximation for l=1
BiBe_ = - 1/2*( np.cos(d) * (3* z_1 * (3/z_0**2 - 1) - 3 * (3 - z_1**2) / z_0) 
       + np.sin(d) * ((3 - z_1**2) * (3/z_0**2 - 1) + 9 *z_1/z_0 )) /(
        np.cos(d) * 3 * z_1 + np.sin(d) * (3-z_1**2))



"Uncomment the following two lines for Bi/Be for perfectly conducting case"

# BiBe = np.array([l/(l+1) for l in range(nlm+1)])
# BiBe_res = np.array([l/(l+1) for l in range(nlm+1)])       

BiBe = np.reshape(BiBe, (nlm+1,1))

phi = np.zeros((2,nlm+1))         
phi[1] = np.array([cmath.phase(BiBe[l]) for l in range (nlm+1)])
phi = np.reshape(phi[1], (nlm+1,1))

if type(sigma_res) is int:
    BiBe_res = np.reshape(BiBe_res, (nlm+1,1))
    phi_res = np.zeros((2,nlm+1))     
    for l in range(nlm+1):
        phi_res[1,l] = cmath.phase(BiBe_res[l])
    phi_res = np.reshape(phi_res[1], (nlm+1,1))

else:
    BiBe_res = np.reshape(BiBe_res, (nlm+1,len(sigma_res),1))
    phi_res = np.zeros((2,nlm+1,len(sigma_res)))     
    for l in range(nlm+1):
        for s in range(len(sigma_res)):
            phi_res[1,l,s] = cmath.phase(BiBe_res[l,s])
    phi_res = np.reshape(phi_res[1], (nlm+1,len(sigma_res),1))

"""
The inducing field is approximated by elliptical polarization in the xy-plane
B_0 is given in Europa IAU, to use EPhiO instead, approximately Bx -> -By, By->Bx
"""

B_0 = np.array([-217, 64])
q_J = -B_0[0]
s_J = -B_0[1]





