# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:36:08 2022

This file solves the mutual induction up to iteration step N and degree/order nlm
Output in the form of internal Gauss Coefficients g[n,l,m], h[n,l,m] for ocean (oc)
and reservoir (res) and radial component of external + internal fields
The following visualizations are implemented in this script
 + Examplary flyby at 25 km altitude
 + Bx-By components in the direct vicinity of the reservoir
 + Mauersberger-Lowes spectrum

"""


import numpy as np
from MagneticField import MagneticField
from Grid import Grid
from Integrator import Integrator
from Trajectory import Trajectory
from LegendrePolynomials import LegendrePolynomials
from CARTGrid import CartGrid
import matplotlib.pyplot as plt
import Parameters as p

nlm = p.nlm #maximum degree/order l,m considered
# nlm_o and nlm_r can be changed to a number <= nlm manually, e.g., 
nlm_o = 2
nlm_r = 2
t = np.pi/p.omega_m # Timepoint for which induced fields are calculated
grid1 = Grid(p.NTH, p.NLAM, (0, np.pi), (0, 2*np.pi)) #Grid Setup
mag1 = MagneticField(p.NTH, p.NLAM, nlm, p.omega_m) #Magnetic Field Class
bindo = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, p.phi[1], t)
bind_sc = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, 0, t)
qjo = -bindo[0]
sjo = -bindo[1]

bindr = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, p.phi_res[1], t)
qjr = -bindr[0]
sjr = -bindr[1]


"""
N sets up the maximum iteration number. For scenarios considered in this work,
i.e. a global ocean and a small, local reservoir, N = 2 suffices as the coupling
from the ocean to the reservoir can be neglected.
"""

N = 2  


"""
Internal Gauss coefficients g[n,l,m] and h[n,l,m] for both ocean (oc) and reservoir (res)
g[0,1,1] and h[0,1,1] are the induction response to thee Jovian background field
"""

g_oc = np.zeros((N,nlm+1,nlm+1))
h_oc = np.zeros((N,nlm+1,nlm+1))

g_res = np.zeros((N,nlm+1,nlm+1))
h_res = np.zeros((N,nlm+1,nlm+1))

g_oc[0,1,1] = np.abs(p.BiBe[1])*qjo
h_oc[0,1,1] = np.abs(p.BiBe[1])*sjo

g_res[0,1,1] = np.abs(p.BiBe_res[1])*qjr
h_res[0,1,1] = np.abs(p.BiBe_res[1])*sjr


"""
The next step sets up the associated Legendre Polynomials, as well as the 
Integrator class 
"""

Legendre = LegendrePolynomials(grid1, nlm) 

P, dP = Legendre.baseline()

grid1.calculate_int_constants(nlm, P)

grid1.transform(p.r_c, p.r_0, p.r_res, p.th_c, p.lam_c) #Calls transformed coordinates
P_tr_o, P_tr_o_dif = Legendre.transformedL(grid1.th_oc) #res surface from oc frame
P_tr_r, P_tr_r_dif = Legendre.transformedL(grid1.th_res) #oc surface from res frame


B_int_r = np.zeros((p.NTH, p.NLAM))
B_int_o = np.zeros((p.NTH, p.NLAM))
integ = Integrator(nlm, p.dth, p.dlam)

def GetGauss(q_J_o, s_J_o, q_J_r, s_J_r, N, Z):
    """
    This function calculates the internal Gauss coefficients for a given
    iteration step and multipole degree

    Parameters
    ----------
    q_J_o : float
        External Gauss coefficient (Bx) of the Jovian background field for ocean.
    s_J_o : TYPE
        External Gauss coefficient (By) of the Jovian background field for ocean.
    q_J_r : TYPE
        External Gauss coefficient (Bx) of the Jovian background field for res.
    s_J_r : TYPE
        External Gauss coefficient (By) of the Jovian background field for res.
    N : TYPE
        Iteration step considered.
    Z : TYPE
        Multipole degree l of the inducing field. This only plays a role for N>2,
        as the inducing field of iteration step n=1 is a pure dipole (before transformation)

    Returns
    -------
    G : array
        Internal Gauss coefficients of of ocean G[0] and reservoir G[1]
        of multipole degree l.
    H : array
        Internal Gauss coefficients of of ocean G[0] and reservoir G[1]
        of multipole degree l.

    """
    #print(Z)
    G = np.zeros((2,nlm+1,nlm+1))
    H = np.zeros((2,nlm+1,nlm+1))
    G[0,1,1] = np.abs(p.BiBe[1]) * q_J_o
    H[0,1,1] = np.abs(p.BiBe[1]) * s_J_o
    G[1,1,1] = np.abs(p.BiBe_res[1]) * q_J_r
    H[1,1,1] = np.abs(p.BiBe_res[1]) * s_J_r
    for _n_ in range(N):
        if _n_ == 0:
        #transform to surface
            b_tr_o = mag1.InternalTransformed(1, grid1, G[0], H[0], p.r_0, grid1.d_oc, grid1.th_oc, grid1.lam_oc, P_tr_o, P_tr_o_dif)
            b_tr_o[np.isnan(b_tr_o)] = 0
            b_tr_r = mag1.InternalTransformed(1, grid1, G[1], H[1], p.r_res, grid1.d_res, grid1.th_res, grid1.lam_res, P_tr_r, P_tr_r_dif)
            b_tr_r[np.isnan(b_tr_r)] = 0
        #perform integration to get external coefficients
        else:
            b_tr_o = mag1.calculate_int_l(int(Z[_n_-1]), grid1, G[0], H[0], p.r_0, grid1.d_oc, grid1.th_oc, grid1.lam_oc, P_tr_o, P_tr_o_dif)
            b_tr_o[np.isnan(b_tr_o)] = 0
            b_tr_r = mag1.calculate_int_l(int(Z[_n_-1]), grid1, G[1], H[1], p.r_res, grid1.d_res, grid1.th_res, grid1.lam_res, P_tr_r, P_tr_r_dif)
            b_tr_r[np.isnan(b_tr_o)] = 0
        integration_result_r = integ.integrate(grid1, b_tr_o)
        integration_result_o = integ.integrate(grid1, b_tr_r)
        Q_res = integration_result_r[0]
        S_res = integration_result_r[1]
        Q_oc = integration_result_o[0]
        S_oc = integration_result_o[1]
        #use Q-response to get internal coefficients
        G[0] = np.abs(p.BiBe) * Q_oc
        H[0] = np.abs(p.BiBe) * S_oc
        G[1] = np.abs(p.BiBe_res) * Q_res
        H[1] = np.abs(p.BiBe_res) * S_res 
    return G, H


count = 0 
phi_o = np.zeros(nlm+1)
phi_r = np.zeros(nlm+1)

"""
This (nested) loop solves the iterative induction coupling. The outer loop
represents the iteration steps n (for N=2 this loop only has one step).
The inner loop runs over the multipole degree l of the inducing field (for n=1).
As the phase shift phi varies with degree l, each degree must be considered individually.
"""
for n in range(1,N):
    print('Calculating internal field for iteration n = {}'.format(n))
    ###finite###
    if n == 1:
        for l in range(1,nlm+1):
            phi_r[l] = p.phi[l] + p.phi_res[1]
            phi_o[l] = p.phi_res[l] + p.phi[1]
            q_J_oc = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_o[count+l], t)[0]
            s_J_oc = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_o[count+l], t)[1]
            q_J_res = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_r[count+l], t)[0]
            s_J_res = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_r[count+l], t)[1]
            GG = GetGauss(q_J_oc, s_J_oc, q_J_res, s_J_res, n, [1,l])
            g_oc[n,l] += GG[0][0][l]
            h_oc[n,l] += GG[1][0][l]
            g_res[n,l] += GG[0][1][l]
            h_res[n,l] += GG[1][1][l]


    else:
        L = np.zeros((n,nlm**n+1))
        count += nlm**(n-1) 
        phi_r = np.append(phi_r,np.zeros(nlm**n))
        phi_o = np.append(phi_o,np.zeros(nlm**n))
        print(len(phi_r))
        for l in range(1,nlm**n+1):
            L[0,l] = int((l-1)/(nlm**(n-1)))+1
            w = (l-1)%nlm
            v = (l-1)/nlm
            for n_ in range(0,n):
                if 0 < n_ < n-1:
                    L[n_,l] = int((l-1)/(nlm**(n - (n_+1))))+1
                    for j in range(1, nlm+1):
                        L[L%nlm == j] = j
                        L[L%nlm == 0] = nlm
                elif n_ == n-1:
                    L[n_,l] = w+1
            if (n%2) == 0:
                phi_r[count+l] = phi_r[count-(nlm)**(n-1)+int(v)+1] + p.phi_res[w+1]
                phi_o[count+l] = phi_o[count-(nlm)**(n-1)+int(v)+1] + p.phi[w+1]
            if (n%2) == 1:
                phi_r[count+l] = phi_r[count-(nlm)**(n-1)+int(v)+1] + p.phi[w+1]
                phi_o[count+l] = phi_o[count-(nlm)**(n-1)+int(v)+1] + p.phi_res[w+1]
            q_J_oc = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_o[count+l], t)[0]
            s_J_oc = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_o[count+l], t)[1]
            q_J_res = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_r[count+l], t)[0]
            s_J_res = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_r[count+l], t)[1]
            GG = GetGauss(q_J_oc, s_J_oc, q_J_res, s_J_res, n, L[:,l])
            g_oc[n,w+1] += GG[0][0][w+1]
            h_oc[n,w+1] += GG[1][0][w+1]
            g_res[n,w+1] += GG[0][1][w+1]
            h_res[n,w+1] += GG[1][1][w+1]

"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


"""
The following example visualizes a flyby at 25 km altitude alongthe y-axis.
The trajectory  must be given in both reservoir and ocean-centered coordinates.
"""

lim_x_r = [p.r_res+25e3, p.r_res+25e3]
lim_y_r = [-200e3, 200e3]
lim_z_r = [0,0]


lim_x_o = [p.r_m+25e3, p.r_m+25e3]


N_pt = 500 #Number of points along the trajectory

trajectory_r = Trajectory(N_pt, lim_x_r[0], lim_x_r[1], lim_y_r[0], lim_y_r[1], lim_z_r[0], lim_z_r[1])
trajectory_o = Trajectory(N_pt, lim_x_o[0], lim_x_o[1], lim_y_r[0], lim_y_r[1], lim_z_r[0], lim_z_r[1])


x_r, y_r, z_r = trajectory_r.cart_coords()
sph_r = trajectory_r.sph_coords()
P_fly_r = Legendre.Legendre_array(N_pt, sph_r[1])

x_o, y_o, z_o = trajectory_o.cart_coords()
sph_o = trajectory_o.sph_coords()
P_fly_o = Legendre.Legendre_array(N_pt, sph_o[1])

###Calculate coupling, superposition, and individuals###

b_0_o = mag1.int_flyby(N_pt, nlm, g_oc[0], h_oc[0], p.r_0, sph_o[0], sph_o[1], sph_o[2], P_fly_o[0], P_fly_o[1])
b_0_r = mag1.int_flyby(N_pt, nlm, g_res[0], h_res[0], p.r_res, sph_r[0], sph_r[1], sph_r[2], P_fly_r[0], P_fly_r[1])
b_ind = np.zeros((3,N_pt))
b_ind[0] = bind_sc[0] * np.sin(sph_o[1]) * np.cos(sph_o[2]) + bind_sc[1] * np.sin(sph_o[1]) * np.sin(sph_o[2])
b_ind[1] = bind_sc[0] * np.cos(sph_o[1]) * np.cos(sph_o[2]) + bind_sc[1] * np.cos(sph_o[1]) * np.sin(sph_o[2])
b_ind[2] = -bind_sc[0] * np.sin(sph_o[2]) + bind_sc[1] *  np.cos(sph_o[2])

b_sp_r = b_0_r + b_0_o


#overall coupling#
b_fly = np.zeros((3,N_pt))
for n in range(N):
    if n == 0:
        b_sp = (mag1.int_flyby(N_pt, nlm_o, g_oc[0], h_oc[0], p.r_0, sph_o[0], sph_o[1], sph_o[2], P_fly_o[0], P_fly_o[1])[0] + 
                mag1.int_flyby(N_pt, nlm_r, g_res[0], h_res[0], p.r_res, sph_r[0], sph_r[1], sph_r[2], P_fly_r[0], P_fly_r[1])[0] )
    b_fly[0] += (mag1.int_flyby(N_pt, nlm_o, g_oc[n], h_oc[n], p.r_0, sph_o[0], sph_o[1], sph_o[2], P_fly_o[0], P_fly_o[1])[0] + 
            mag1.int_flyby(N_pt, nlm_r, g_res[n], h_res[n], p.r_res, sph_r[0], sph_r[1], sph_r[2], P_fly_r[0], P_fly_r[1])[0] )

    
fig, ax = plt.subplots(1,1,figsize=(10,4))


ax.plot(y_r/1e3, b_0_o[0] + b_ind[0], label = r'Ocean induction')
ax.plot(y_r/1e3, b_fly[0] + b_ind[0], color ='k', label = r'Coupled induction')
ax.plot(y_r/1e3, b_sp + b_ind[0], label = r'Superposition')
ax.set_xlim(-200,200)
ax.set_ylabel(r'$B_r$ /nT', fontsize = 14)
ax.set_xlabel(r'$y$ /km', fontsize = 14)
ax.tick_params('both', labelsize = 13)
plt.legend(fontsize = 14,frameon=False)
#plt.savefig('Flyby.pdf', bbox_inches='tight')


"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

"""
This example visualizes the magnetic field components in the xy-plane.
For this, the area in which the fields are to be calculated must be specified,
e.g. in the direct vicinity around the reservoir.
"""

N_pt = 500
lim_x = [1480e3, 1600e3]  
lim_y = [-60e3, 60e3]


cart_grid = CartGrid(N_pt, N_pt, lim_x, lim_y, 0)

r, th, lam = cart_grid.transform_2_sph()
R, LAM = cart_grid.sph_mesh()
R2, LAM2 = cart_grid.sph_mesh_2()

P = Legendre.Legendre_array(N_pt, th)
magnetic1 = MagneticField(N_pt, N_pt, nlm, p.omega_m)

B_ind = np.ones((2,N_pt,N_pt))
B_ind[0] = B_ind[0]*bind_sc[0]
#B_ind[1] = -B_ind[1]*p.s_J


fullB_res = np.zeros((2, N_pt, N_pt))
fullB_oc = np.zeros((2, N_pt, N_pt))



for n in range(N):
    if n == 0:
        fullB_res += magnetic1.int_streamline(N_pt, 1, g_res[n], h_res[n], p.r_res, R2, th, LAM2, P[0], P[1])
        fullB_oc += magnetic1.int_streamline(N_pt, 1, g_oc[n], h_oc[n], p.r_0, R, th, LAM, P[0], P[1])
        dipB_res = magnetic1.int_streamline(N_pt, 1, g_res[n], h_res[n], p.r_res, R2, th, LAM2, P[0], P[1])
        dipB_oc = magnetic1.int_streamline(N_pt, 1, g_oc[n], h_oc[n], p.r_0, R, th, LAM, P[0], P[1])
    else:
        fullB_res += magnetic1.int_streamline(N_pt, nlm_r, g_res[n], h_res[n], p.r_res, R2, th, LAM2, P[0], P[1])
        #fullB_oc += magnetic1.int_streamline(N_pt, nlm_o, g_oc[n], h_oc[n], p.r_0, R, th, LAM, P[0], P[1])

for i in range(N_pt):
    for j in range(N_pt):
        if R[i,j] <= p.r_0:
            fullB_oc[0][i,j] = 0
            fullB_res[0][i,j] = 0
            fullB_oc[1][i,j] = 0
            fullB_res[1][i,j] = 0
            dipB_oc[0][i,j] = 0
            dipB_res[0][i,j] = 0
            dipB_oc[1][i,j] = 0
            dipB_res[1][i,j] = 0
            B_ind[1][i,j] = 0
            B_ind[0][i,j] = 0

        if R2[i,j] <= p.r_res:
            fullB_oc[0][i,j] = 0
            fullB_res[0][i,j] = 0
            fullB_oc[1][i,j] = 0
            fullB_res[1][i,j] = 0
            dipB_oc[0][i,j] = 0
            dipB_res[0][i,j] = 0
            dipB_oc[1][i,j] = 0
            dipB_res[1][i,j] = 0
            B_ind[1][i,j] = 0
            B_ind[0][i,j] = 0

B_tot = np.sqrt((fullB_oc[0]+fullB_res[0]+B_ind[0])**2 + (fullB_oc[1]+fullB_res[1]+B_ind[1])**2)
B_tot[np.isnan(B_tot)] = 0
step = 25
extent = [lim_x[0]/1e3,lim_x[1]/1e3,lim_y[0]/1e3,lim_y[1]/1e3]
U = (fullB_oc[0]+fullB_res[0]+B_ind[0])/B_tot
V = (fullB_oc[1]+fullB_res[1]+B_ind[1])/B_tot
norm = plt.Normalize(0, 50)
fig, ax = plt.subplots(figsize=(9,9))



q = ax.quiver(cart_grid.X[::step,::step]/1e3, cart_grid.Y[::step,::step]/1e3, U[::step,::step], V[::step,::step],color ='k',units='xy')
matrix = ax.imshow(B_tot, norm=norm, cmap = 'autumn', extent = extent,origin = 'lower')
cbar = fig.colorbar(matrix, fraction=0.046, pad=0.04)
cbar.set_label(label=r'$B_{tot}$ /nT', size = 16)
cbar.ax.tick_params(labelsize=14)
ax.plot(p.r_0/1e3 * np.cos(np.linspace(0,2*np .pi, num=360)), p.r_0/1e3 * np.sin(np.linspace(0,2*np.pi, num=360)), color = 'k')
ax.plot(p.r_c/1e3+p.r_res/1e3 * np.cos(np.linspace(0,2*np.pi)), p.r_res/1e3 * np.sin(np.linspace(0,2*np.pi)), color = 'k')
ax.plot(p.r_m/1e3 * np.cos(np.linspace(0,2*np .pi, num=360)), p.r_m/1e3 * np.sin(np.linspace(0,2*np.pi, num=360)), color = 'k', linestyle = 'dashed')
ax.set_xlabel(r"$x$ /km", fontsize=16)
ax.set_ylabel(r"$y$ /km", fontsize=16)
ax.tick_params('both', labelsize=15)
ax.set_xlim(1480, 1600)
ax.set_ylim(-60,60)


ax.text(1540,7,'Res', fontsize=17)
ax.text(1520,-29, 'Icy Crust', fontsize = 17)
ax.text(1492,0,'Ocean',fontsize=17)

"Display orientation of dipole field in x-y plane"
plt.arrow(1540, 0, bind_sc[0]/12, bind_sc[1]/12, width = 0.5,head_width=2, color = 'k', label = 'B_{ind}')
plt.arrow(1540, 0, (g_res[1,1,1]+g_res[0,1,1])/6, (h_res[1,1,1]+h_res[0,1,1])/6, width = 0.5,head_width=2, color = 'cyan')
plt.arrow(1540, 0, g_oc[0,1,1]/6, h_oc[0,1,1]/6, width = 0.5,head_width=2, color = 'g')
#plt.savefig('BField_Arrow_finite.pdf')

"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
"""
This example calculates the Mauersberger spectrum of the first and second iteration
for both ocean and reservoir.
"""

# MAUERSBERGER-LOWES SPECTRUM

R = np.zeros((N,nlm+1))
R_r = np.zeros((N,nlm+1))
fig, ax = plt.subplots(figsize=(10,5))
for n in range(N):
    for l in range(1,nlm+1):
        for m in range(0, l+1):
            R[n,l] += (l+1)*((g_oc[n,l,m])**2 + (h_oc[n,l,m])**2)
            R_r[n,l] += (l+1)*((g_res[n,l,m])**2 + (h_res[n,l,m])**2)
ax.scatter(range(1,2), R[0,1], color ='k', marker = '^', s = 25, label = r'Ocean ($n=1$)')
ax.scatter(range(nlm+1), R[1], color = 'k', edgecolor = 'k', marker = 'v', s = 25, label = r'Ocean ($n=2$)')
ax.scatter(range(1,2), R_r[0,1], color ='violet', marker = '^', s = 25, label = r'Reservoir ($n=1$)')
ax.scatter(range(nlm+1), R_r[1], color = 'violet', edgecolor = 'violet', marker = 'v', s = 25, label = r'Reservoir ($n=2$)')
ax.set_yscale('log')
ax.set_ylim(1e-9,1e5)
ax.set_xlim(0,nlm)
ax.set_xlabel(r'Degree $l$', fontsize = 14)
ax.set_ylabel(r'$R$ /nT$^2$', fontsize = 14)
ax.tick_params('both', labelsize=13)
plt.legend(frameon=False)
#plt.savefig('Mauersberger.pdf',bbox_inches='tight')