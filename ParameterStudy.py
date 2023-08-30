# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:36:08 2022

Like general.py, this script solves the mutual induction between reservoir and 
ocean, but does so for the full array of considered conductivities. This script 
can be used to obtain the results that were shown in Figure 8

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

# This script outputs the maximum deviation between radial components with
# and without reservoir for full conductivity range

nlm = p.nlm #maximum degree/order l,m considered
nlm_o = 4
nlm_r = 4
t = np.pi/(p.omega_m)
grid1 = Grid(p.NTH, p.NLAM, (0, np.pi), (0, 2*np.pi)) #Grid Setup!!!
mag1 = MagneticField(p.NTH, p.NLAM, nlm, p.omega_m) #Magnetic Field Class!!!

Legendre = LegendrePolynomials(nlm) #Legendre Class!!!
C, S = Legendre.baseline(grid1)
grid1.calculate_int_constants(nlm, C, S)
grid1.transform(p.r_c, p.r_0, p.r_res, p.th_c, p.lam_c) #Calls transformed coordinates
P_tr_o, P_tr_o_dif = Legendre.transformed_Legendre(grid1, nlm, grid1.th_oc, grid1.lam_oc) #res surface from oc frame
P_tr_r, P_tr_r_dif = Legendre.transformed_Legendre(grid1, nlm, grid1.th_res, grid1.lam_res) #oc surface from res frame

N = 2    #number of iterations
K = 2 #0 superconduct, 1 finite

def GetGauss(q_J_o, s_J_o, q_J_r, s_J_r, N, Z):
    #print(Z)
    G = np.zeros((2,nlm+1,nlm+1))
    H = np.zeros((2,nlm+1,nlm+1))
    G[0,1,1] = np.abs(p.BiBe[1]) * q_J_o
    H[0,1,1] = np.abs(p.BiBe[1]) * s_J_o
    G[1,1,1] = np.abs(p.BiBe_res[1,s]) * q_J_r
    H[1,1,1] = np.abs(p.BiBe_res[1,s]) * s_J_r
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
        G[1] = np.abs(p.BiBe_res[:,s]) * Q_res
        H[1] = np.abs(p.BiBe_res[:,s]) * S_res 
    return G, H, Q_oc, S_oc, Q_res, S_res, b_tr_o, b_tr_r
deltab_max = np.zeros(len(p.sigma_res))
for  s in range(len(p.sigma_res)):
    print('-----Conductvity:', p.sigma_res[s], 'S/m-----')
    bindo = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, p.phi[1], t)
    bind_sc = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, 0, t)
    qjo = -bindo[0]
    sjo = -bindo[1]
    #print(qjo, sjo)
    bindr = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, p.phi_res[1,s], t)
    qjr = -bindr[0]
    sjr = -bindr[1]
    q_sc = -bind_sc[0]
    s_sc = -bind_sc[1]
    
    ###Set up baseline iteration parameters###
    
    g_oc = np.zeros((N,nlm+1,nlm+1))
    h_oc = np.zeros((N,nlm+1,nlm+1))
    
    g_res = np.zeros((N,nlm+1,nlm+1))
    h_res = np.zeros((N,nlm+1,nlm+1))
    
    q_oc = np.zeros((nlm+1,nlm+1))
    s_oc = np.zeros((nlm+1,nlm+1))
    
    q_res = np.zeros((nlm+1,nlm+1))
    s_res = np.zeros((nlm+1,nlm+1))
    
    g_oc[0,1,1] = np.abs(p.BiBe[1])*qjo
    h_oc[0,1,1] = np.abs(p.BiBe[1])*sjo
    
    g_res[0,1,1] = np.abs(p.BiBe_res[1,s])*qjr
    h_res[0,1,1] = np.abs(p.BiBe_res[1,s])*sjr

    integ = Integrator(nlm, p.dth, p.dlam)
    
    start_time = time.time()
    
    z = 0
    count = 0
    phi_o = np.zeros(nlm+1)
    phi_r = np.zeros(nlm+1)
    for n in range(1,N):
        #print('Calculating internal field for iteration n = {}'.format(n))
        ###finite###
        for k in range(2):
            if k == 1:
                if n == 1:
                    for l in range(1,nlm+1):
                        phi_r[l] = p.phi[l] + p.phi_res[1,s]
                        phi_o[l] = p.phi_res[l,s] + p.phi[1]
                        q_J_oc = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_o[count+l], t)[0]
                        s_J_oc = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_o[count+l], t)[1]
                        q_J_res = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_r[count+l], t)[0]
                        s_J_res = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_r[count+l], t)[1]
                        GG = GetGauss(q_J_oc, s_J_oc, q_J_res, s_J_res, n, [1,l])
                        #g_oc[n,l] += GG[0][0][l]
                        #h_oc[n,l] += GG[1][0][l]
                        g_res[n,l] += GG[0][1][l]
                        h_res[n,l] += GG[1][1][l]
        
    
    lim_x_r = [p.r_res+25e3, p.r_res+25e3]
    lim_y_r = [-1000e3, 1000e3]
    lim_z_r = [0,0]
    
    
    lim_x_o = [p.r_m+25e3, p.r_m+25e3]
    
    
    N_pt = 500
    
    trajectory_r = Trajectory(N_pt, lim_x_r[0], lim_x_r[1], lim_y_r[0], lim_y_r[1], lim_z_r[0], lim_z_r[1])
    trajectory_o = Trajectory(N_pt, lim_x_o[0], lim_x_o[1], lim_y_r[0], lim_y_r[1], lim_z_r[0], lim_z_r[1])
    
    
    x_r, y_r, z_r = trajectory_r.cart_coords()
    sph_r = trajectory_r.sph_coords()
    P_fly_r = Legendre.Legendre_array(N_pt, sph_r[1])
    
    x_o, y_o, z_o = trajectory_o.cart_coords()
    sph_o = trajectory_o.sph_coords()
    P_fly_o = Legendre.Legendre_array(N_pt, sph_o[1])
    
    ###Calculate coupling, superposition, and individuals###
    
    b_0_o = mag1.int_flyby(N_pt, nlm, g_oc[0], h_oc[0], p.r_0, sph_o[0], sph_o[1], sph_o[2], p.phi[1],P_fly_o[0], P_fly_o[1])
    b_0_r = mag1.int_flyby(N_pt, nlm, g_res[0], h_res[0], p.r_res, sph_r[0], sph_r[1], sph_r[2],p.phi_res[1], P_fly_r[0], P_fly_r[1])
    b_ind = np.zeros((3,N_pt))
    b_ind[0] = bind_sc[0] * np.sin(sph_o[1]) * np.cos(sph_o[2]) + bind_sc[1] * np.sin(sph_o[1]) * np.sin(sph_o[2])
    b_ind[1] = bind_sc[0] * np.cos(sph_o[1]) * np.cos(sph_o[2]) + bind_sc[1] * np.cos(sph_o[1]) * np.sin(sph_o[2])
    b_ind[2] = -bind_sc[0] * np.sin(sph_o[2]) + bind_sc[1] *  np.cos(sph_o[2])
    
    b_sp_r = b_0_r + b_0_o
    #b_sp_r_inf = b_0_r_inf + b_0_o_inf
    
    #overall coupling#
    b_fly = np.zeros((3,N_pt))
    for n in range(N):
        b_fly[0] += (mag1.int_flyby(N_pt, nlm_o, g_oc[n], h_oc[n], p.r_0, sph_o[0], sph_o[1], sph_o[2],0, P_fly_o[0], P_fly_o[1])[0] + 
                mag1.int_flyby(N_pt, nlm_r, g_res[n], h_res[n], p.r_res, sph_r[0], sph_r[1], sph_r[2], 0, P_fly_r[0], P_fly_r[1])[0] )
        b_fly[1] += (mag1.int_flyby(N_pt, nlm_o, g_oc[n], h_oc[n], p.r_0, sph_o[0], sph_o[1], sph_o[2],0, P_fly_o[0], P_fly_o[1])[1] + 
                mag1.int_flyby(N_pt, nlm_r, g_res[n], h_res[n], p.r_res, sph_r[0], sph_r[1], sph_r[2], 0, P_fly_r[0], P_fly_r[1])[1] )
        b_fly[2] += (mag1.int_flyby(N_pt, nlm_o, g_oc[n], h_oc[n], p.r_0, sph_o[0], sph_o[1], sph_o[2],0, P_fly_o[0], P_fly_o[1])[2] + 
                mag1.int_flyby(N_pt, nlm_r, g_res[n], h_res[n], p.r_res, sph_r[0], sph_r[1], sph_r[2], 0, P_fly_r[0], P_fly_r[1])[2] )
    # fig, ax = plt.subplots(3,1,figsize=(9,7))
    
    # for j in range(0,3):
    #     ax[j].plot(x_r/1e3, b_0_o[j]+b_ind[j])
    #     ax[j].plot(x_r/1e3, b_fly[j]+b_ind[j], color ='k')
    #     ax[j].set_xlim(-800,800)
    # ax[0].set_ylabel(r'$B_r$ /nT')
    # ax[1].set_ylabel(r'$B_\theta$ /nT')
    # ax[2].set_ylabel(r'$B_\lambda$ /nT')
    # ax[0].set(xticklabels=[])
    # ax[1].set(xticklabels=[])
    # ax[2].set_xlabel(r'$x$ /km')
    
    # plt.show()
    deltab_max[s] = abs(b_fly[0]-b_0_o[0]).max()
    print('Maximum difference:', f'{deltab_max[s]:.5f}' ,'nT')

print('Full array for radius', p.r_res/1e3, 'km')
print(deltab_max)
