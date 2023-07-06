# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:36:08 2022

@author: Jason
"""

import numpy as np
from MagneticField import MagneticField
from Grid import Grid
from Integrator import Integrator
from Trajectory import Trajectory
from LegendrePolynomials import LegendrePolynomials
import matplotlib.pyplot as plt
import Parameters as p
import sys
import time
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Plot magnetic field lines in perfectly conducting case
# Remember to change to perfectly conducting in parameters module

nlm = p.nlm #maximum degree/order l,m considered
nlm_o = 5
nlm_r = 5
t = np.pi/(2*p.omega_m)
grid1 = Grid(p.NTH, p.NLAM, (0, np.pi), (0, 2*np.pi)) #Grid Setup!!!
mag1 = MagneticField(p.NTH, p.NLAM, nlm, p.omega_m) #Magnetic Field Class!!!
bindo = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, p.phi[1], t)
bind_sc = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, 0, t)
qjo = -bindo[0]
sjo = -bindo[1]
print(qjo, sjo)
bindr = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, p.phi_res[1], t)
qjr = -bindr[0]
sjr = -bindr[1]
q_sc = -bind_sc[0]
s_sc = -bind_sc[1]
###Set up baseline iteration parameters###


N = 2 #number of iterations
K = 2 #0 superconduct, 1 finite

g_oc = np.zeros((N,nlm+1,nlm+1))
h_oc = np.zeros((N,nlm+1,nlm+1))

g_res = np.zeros((N,nlm+1,nlm+1))
h_res = np.zeros((N,nlm+1,nlm+1))

q_oc = np.zeros((nlm+1,nlm+1))
s_oc = np.zeros((nlm+1,nlm+1))

q_res = np.zeros((nlm+1,nlm+1))
s_res = np.zeros((nlm+1,nlm+1))

###Seed starting parameters###

g_oc[0,1,1] = np.abs(p.BiBe[1])*qjo
h_oc[0,1,1] = np.abs(p.BiBe[1])*sjo

g_res[0,1,1] = np.abs(p.BiBe_res[1])*qjr
h_res[0,1,1] = np.abs(p.BiBe_res[1])*sjr

###Start looping###

Legendre = LegendrePolynomials(nlm) #Legendre Class!!!


# print('Load main spherical harmonics...')
# SPH = np.loadtxt('SphericalHarmonics40.txt')
C, S = Legendre.baseline(grid1)
# C = np.reshape(C,(41,41,grid1.nth,grid1.nlam))
# S = np.reshape(S,(41,41,grid1.nth,grid1.nlam))
grid1.calculate_int_constants(nlm, C, S)

grid1.transform(p.r_c, p.r_0, p.r_res, p.th_c, p.lam_c) #Calls transformed coordinates
P_tr_o, P_tr_o_dif = Legendre.transformed_Legendre(grid1, nlm, grid1.th_oc, grid1.lam_oc) #res surface from oc frame
P_tr_r, P_tr_r_dif = Legendre.transformed_Legendre(grid1, nlm, grid1.th_res, grid1.lam_res) #oc surface from res frame


integ = Integrator(nlm, p.dth, p.dlam)


start_time = time.time()
def GetGauss(q_J_o, s_J_o, q_J_r, s_J_r, N, Z):
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
            b_tr_o = mag1.calculate_int(1, grid1, G[0], H[0], p.r_0, grid1.d_oc, grid1.th_oc, grid1.lam_oc, P_tr_o, P_tr_o_dif)
            b_tr_o[np.isnan(b_tr_o)] = 0
            b_tr_r = mag1.calculate_int(1, grid1, G[1], H[1], p.r_res, grid1.d_res, grid1.th_res, grid1.lam_res, P_tr_r, P_tr_r_dif)
        #b_tr_o = mag1.int_test(G[1], H[1], p.phi_res[1], C, S)
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
    return G, H, Q_oc, S_oc, Q_res, S_res, b_tr_o, b_tr_r

z = 0
count = 0
phi_o = np.zeros(nlm+1)
phi_r = np.zeros(nlm+1)
for n in range(1,N):
    print('Calculating internal field for iteration n = {}'.format(n))
    ###finite###
    for k in range(2):
        if k == 1:
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
                    # q_oc[n-1,w+1] += GG[2]
                    # s_oc[n-1,w+1] += GG[3]
                    # q_res[n-1,w+1] += GG[4]
                    # s_res[n-1,w+1] += GG[5]
                    g_oc[n,w+1] += GG[0][0][w+1]
                    h_oc[n,w+1] += GG[1][0][w+1]
                    g_res[n,w+1] += GG[0][1][w+1]
                    h_res[n,w+1] += GG[1][1][w+1]
                


N= 2
N_pt = 1000
lim_x = [-140e3, 140e3]
lim_y = [p.r_m-100e3, p.r_m+100e3]
lim_z = [0,0]
from CARTGrid import CartGrid
cart_grid = CartGrid(N_pt, N_pt, N_pt, lim_x, lim_y, lim_z)



r, th, lam = cart_grid.transform_2_sph()
R, LAM = cart_grid.sph_mesh()
R2, LAM2 = cart_grid.sph_mesh_2()
Legendre = LegendrePolynomials(nlm)
P = Legendre.Legendre_array(N_pt, th)
magnetic1 = MagneticField(N_pt, N_pt, nlm, p.omega_m)

B_ind = np.ones((2,N_pt,N_pt))
B_ind[0] = -B_ind[0]*0
B_ind[1] = -B_ind[1]*p.s_J


fullB_res = np.zeros((4, N_pt, N_pt))
fullB_oc = np.zeros((4, N_pt, N_pt))



for n in range(N):
    if n == 0:
        fullB_res += magnetic1.int_streamline(N_pt, 1, g_res[n], h_res[n], p.r_res, R2, th, LAM2, P[0], P[1])
        fullB_oc += magnetic1.int_streamline(N_pt, 1, g_oc[n], h_oc[n], p.r_0, R, th, LAM, P[0], P[1])
        dipB_res = magnetic1.int_streamline(N_pt, 1, g_res[n], h_res[n], p.r_res, R2, th, LAM2, P[0], P[1])
        dipB_oc = magnetic1.int_streamline(N_pt, 1, g_oc[n], h_oc[n], p.r_0, R, th, LAM, P[0], P[1])
    else:
        print('Youre here')
        fullB_res += magnetic1.int_streamline(N_pt, nlm, g_res[n], h_res[n], p.r_res, R2, th, LAM2, P[0], P[1])
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

B_tot = np.sqrt((fullB_oc[0]+fullB_res[0])**2 + (fullB_oc[1]+fullB_res[1]+B_ind[1])**2)
B_tot_sp = np.sqrt((dipB_oc[0]+dipB_res[0])**2 + (dipB_oc[1]+dipB_res[1]+B_ind[1])**2)
B_tot[np.isnan(B_tot)] = 0
norm = plt.Normalize(0, 50)
fig, ax = plt.subplots(1,2,figsize=(10,10))
levels = np.arange(25,65,step=5)
pp = PdfPages('Test.pdf')
#ax.streamplot(cart_grid.X/1e3, cart_grid.Y/1e3, B_ind[0]+fullB_res[0]+fullB_oc[0], B_ind[1]+fullB_res[1]+fullB_oc[1], density=[1], color = 'k')
CS_r = ax[0].streamplot(cart_grid.X/1e3, cart_grid.Y/1e3, dipB_res[0]+dipB_oc[0], dipB_oc[1]+dipB_res[1]+B_ind[1],density=[1], color ='k')
CS = ax[1].streamplot(cart_grid.X/1e3, cart_grid.Y/1e3, fullB_oc[0]+fullB_res[0], fullB_res[1]+fullB_oc[1]+B_ind[1],density=[1], color ='k')
matrix_sp =  ax[0].imshow(B_tot_sp, norm=norm, cmap = 'autumn', extent = [-140,140,lim_y[0]/1e3,lim_y[1]/1e3],origin = 'lower')
matrix = ax[1].imshow(B_tot, norm=norm, cmap = 'autumn', extent = [-140,140,lim_y[0]/1e3,lim_y[1]/1e3],origin = 'lower')
fig.subplots_adjust(right=0.8)
cbarax = fig.add_axes([0.85,0.32,0.03,0.36])
cbar = fig.colorbar(matrix, cax= cbarax)
cbar.set_label(label=r'$B_{tot}$ /nT', size = 16)
ax[0].set_ylabel(r"$y$ /km", fontsize=16)
cbar.ax.tick_params(labelsize=14)
for i in range(2):
    ax[i].plot(p.r_0/1e3 * np.cos(np.linspace(0,2*np .pi, num=360)), p.r_0/1e3 * np.sin(np.linspace(0,2*np.pi, num=360)), color = 'k')
    ax[i].plot(p.r_res/1e3 * np.cos(np.linspace(0,2*np.pi)), p.r_c/1e3+p.r_res/1e3 * np.sin(np.linspace(0,2*np.pi)), color = 'k')
    ax[i].plot(p.r_m/1e3 * np.cos(np.linspace(0,2*np .pi, num=360)), p.r_m/1e3 * np.sin(np.linspace(0,2*np.pi, num=360)), color = 'k', linestyle = 'dotted')
    ax[i].set_xlabel(r"$x$ /km", fontsize=16)
    ax[i].tick_params('both', labelsize=15)
    ax[i].set_xlim(-60,60)
    ax[i].set_ylim(1480, 1600)
    ax[i].text(2,1540,'Res', fontsize=17)
    ax[i].text(-12,1492,'Ocean',fontsize=17)
    ax[i].arrow(0, 1540, bind_sc[0], bind_sc[1]/12, width = 0.5,head_width=2, color = 'k', label = 'B_{ind}')
    ax[i].arrow(0,1540, (g_res[1,1,1]+g_res[0,1,1])/6, (h_res[1,1,1]+h_res[0,1,1])/6, width = 0.5,head_width=2, color = 'cyan')
    ax[i].arrow(0,1540, g_oc[0,1,1]/6, h_oc[0,1,1]/6, width = 0.5,head_width=2, color = 'g')
ax[1].set(yticklabels=[])
ax[0].set_title('Superposition',fontsize=16)
ax[1].set_title('Mutual Induction Coupling',fontsize=16)
#ax.clabel(CS_r)
#ax.set_yticks([p.r_c/1e3-100,p.r_c/1e3-50,p.r_c/1e3,p.r_c/1e3+50], [-100, -50, 0, 50])
#plt.title(r'Magnetic field lines in the $x-y-$plane')

#plt.savefig('BFieldLines_SC.pdf', bbox_inches='tight')

