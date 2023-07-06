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

# This was used to create Mauersberger spectrum and field lines in finite
# conducting case

nlm = p.nlm #maximum degree/order l,m considered
nlm_o = 4
nlm_r = 4
t = np.pi/(2*p.omega_m)
grid1 = Grid(p.NTH, p.NLAM, (0, np.pi), (0, 2*np.pi)) #Grid Setup!!!
mag1 = MagneticField(p.NTH, p.NLAM, nlm, p.omega_m, p.t) #Magnetic Field Class!!!
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


N = 2  #number of iterations
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

b_tr_o = np.zeros((p.NTH,p.NLAM))
b_tr_r = np.zeros((p.NTH,p.NLAM))


b_ext_o = np.zeros((N,p.NTH,p.NLAM))
b_ext_r = np.zeros((N,p.NTH,p.NLAM))

R_r = np.zeros((N,nlm+1))
R_o = np.zeros((N,nlm+1))

B_int_r = np.zeros((p.NTH, p.NLAM))
B_int_o = np.zeros((p.NTH, p.NLAM))
integ = Integrator(nlm, p.dth, p.dlam)

B_ext_r = np.zeros((p.NTH,p.NLAM))
B_ext_o = np.zeros((p.NTH, p.NLAM))
k_2_far_r = np.zeros((K+1,N))
k_2_near_r = np.zeros((K+1,N))
k_2_far_o = np.zeros((K+1,N))
k_2_near_o = np.zeros((K+1,N))

k_2_far_r[1,0] = 1
k_2_near_r[1,0] = 1
k_2_far_o[1,0] = 1
k_2_near_o[1,0] = 1

B_ext_r += mag1.calculate_int(1, grid1, g_oc[0], h_oc[0], p.r_0, grid1.d_oc, grid1.th_oc, grid1.lam_oc, P_tr_o, P_tr_o_dif)
b_ext_r[0] = mag1.calculate_int(1, grid1, g_oc[0], h_oc[0], p.r_0, grid1.d_oc, grid1.th_oc, grid1.lam_oc, P_tr_o, P_tr_o_dif)
B_int_r += mag1.int_test(1,g_res[0], h_res[0], C, S)

B_ext_o += mag1.calculate_int(1, grid1, g_res[0], h_res[0], p.r_res, grid1.d_res, grid1.th_res, grid1.lam_res, P_tr_r, P_tr_r_dif)
b_ext_o[0] = mag1.calculate_int(1, grid1, g_res[0], h_res[0], p.r_res, grid1.d_res, grid1.th_res, grid1.lam_res, P_tr_r, P_tr_r_dif)
B_int_o += mag1.int_test(1,g_oc[0], h_oc[0], C, S)

b_ind_0_r = mag1.int_test(1, g_res[0], h_res[0], C, S)
b_ind_0_o = mag1.int_test(1, g_oc[0], h_oc[0], C, S)

k_2_far_r[2,0]  = B_int_r[90,90] + B_ext_r[90,90]
k_2_near_r[2,0] = B_int_r[90,270] + B_ext_r[90,270]

k_2_near_o[2,0]  = B_int_o[90,90] + B_ext_o[90,90]
k_2_far_o[2,0] = B_int_o[90,270] + B_ext_o[90,270]
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
                    if l == 1:
                        q_oc += GG[2]
                        s_oc += GG[3]
                        q_res += GG[4]
                        s_res += GG[5]
                        b_tr_o += GG[6]
                        b_tr_r += GG[7]
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
                

    # plot_l_o = input('Enter maximum degree for ocean plots: \n')
    # plot_l_r = input('Enter maximum degree for rservoir plots: \n')
    plot_l_o = nlm
    plot_l_r = nlm
    B_ext_r += mag1.calculate_int(nlm_o, grid1, g_oc[n], h_oc[n], p.r_0, grid1.d_oc, grid1.th_oc, grid1.lam_oc, P_tr_o, P_tr_o_dif)
    b_ext_r[n] = mag1.calculate_int(nlm_o, grid1, g_oc[n], h_oc[n], p.r_0, grid1.d_oc, grid1.th_oc, grid1.lam_oc, P_tr_o, P_tr_o_dif)
    B_int_r += mag1.int_test(nlm_r, g_res[n], h_res[n], C, S)
    
    B_ext_o += mag1.calculate_int(nlm_r, grid1, g_res[n], h_res[n], p.r_res, grid1.d_res, grid1.th_res, grid1.lam_res, P_tr_r, P_tr_r_dif)
    b_ext_o[n] = mag1.calculate_int(nlm_r, grid1, g_res[n], h_res[n], p.r_res, grid1.d_res, grid1.th_res, grid1.lam_res, P_tr_r, P_tr_r_dif)
    B_int_o += mag1.int_test(nlm_o,g_oc[n], h_oc[n], C, S)

    k_2_far_r[1,n] = (B_ext_r[90, 90] + B_int_r[90,90])/(b_ext_r[0][90,90] + b_ind_0_r[90,90])
    k_2_near_r[1,n] = (B_ext_r[90, 270] + B_int_r[90,270])/(b_ext_r[0][90,270] + b_ind_0_r[90,270])
    
    k_2_far_r[2,n] = B_int_r[90,90] + B_ext_r[90,90]
    k_2_near_r[2,n] = B_int_r[90,270] + B_ext_r[90,270]
    
    k_2_far_o[1,n] = (B_ext_o[90, 270] + B_int_o[90,270])/(b_ext_o[0][90,270] + b_ind_0_o[90,270])
    k_2_near_o[1,n] = (B_ext_o[90, 90] + B_int_o[90,90])/(b_ext_o[0][90,90] + b_ind_0_o[90,90])
    
    k_2_far_o[2,n] = B_int_o[90,270] + B_ext_o[90,270]
    k_2_near_o[2,n] = B_int_o[90,90] + B_ext_o[90,90]
    
    print('Values for Reservoir Surface points')
    print('Far side superposition:', b_ind_0_r[90,90]+b_ext_r[0][90,90])
    # print('Near side superposition:', b_ind_0_r[90,270]+b_ext_r[0][90,270])

    print('Far side coupling effect:', B_int_r[90,90] + B_ext_r[90,90])
    # print('Near side coupling effect:', B_int_r[90,270] + B_ext_r[90,270])
    print('Dipole amplitude:', 2*np.abs(p.BiBe_res[1]))
    print('Far side effective amplitude:' , 2*np.abs(p.BiBe_res[1]) * B_int_r[90,90]/b_ind_0_r[90,90])
    # print('Near side coupling index:' , k_2_near_r[1,n])
    
    # print('Values for Ocean Surface points')
    # print('Far side superposition:', b_ind_0_o[90,270]+b_ext_o[0][90,270])
    # print('Near side superposition:', b_ind_0_o[90,90]+b_ext_o[0][90,90])

    # print('Far side coupling effect:', B_int_o[90,270] + B_ext_o[90,270])
    # print('Near side coupling effect:', B_int_o[90,90] + B_ext_o[90,90])

    # print('Far side coupling index:' , k_2_far_o[1,n])
    # print('Near side coupling index:' , k_2_near_o[1,n])

    ###CALCULATE DEVIATION AFTER EACH ITERATION###
    # if n > 0:
    #     print('Normalized Deviation after iteration {}'.format(n), np.abs(b_ext_r[n][90,270]/B_ext_r[90,270]))
    #     print('Absolute change in feedback strength, res far', abs(k_2_far_r[2,n] - k_2_far_r[2,n-1]))
    #     print('Absolute change in feedback strength, oc near', abs(k_2_near_o[2,n] - k_2_near_o[2,n-1]))
    #     if abs(k_2_far_r[2,n] - k_2_far_r[2,n-1]) < 1e-2:
    #         print('Sufficient res precision with coppling reached after iteration', n)
    # plt.plot(B_int_r[90, :], label = 'Mutual induction')
    # plt.plot(mag1.int_test(1,g_res[0], h_res[0], C, S)[90, :], label = 'Dipole response')
    # #plt.plot((mag1.int_test_l(g_res[1,n], h_res[1,n], C, S, 2))[90, :], label = 'n2')
    # plt.legend()
    # plt.show()
    # plt.plot(b_tr_o[90, :], label = 'External')
    # plt.plot(mag1.calculate_ext(nlm, q_res, s_res, C, S)[90,:], linestyle='dashed')
    # plt.show()
    # plt.plot(b_tr_r[90, :], label = 'External')
    # plt.plot(mag1.calculate_ext(nlm, q_oc, s_oc, C, S)[90,:], linestyle='dashed')
    # plt.show()
    # plt.plot((b_tr_r- mag1.calculate_ext(nlm, q_oc, s_oc, C, S))[90,:])
    # plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
#Calculate coupling index kappa#

lim_x_r = [-200e3, 200e3]
lim_y_r = [p.r_res+25e3, p.r_res+25e3]
lim_z_r = [0,0]


lim_y_o = [p.r_m+25e3, p.r_m+25e3]


N_pt = 500

trajectory_r = Trajectory(N_pt, lim_x_r[0], lim_x_r[1], lim_y_r[0], lim_y_r[1], lim_z_r[0], lim_z_r[1])
trajectory_o = Trajectory(N_pt, lim_x_r[0], lim_x_r[1], lim_y_o[0], lim_y_o[1], lim_z_r[0], lim_z_r[1])


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
    if n == 0:
        b_sp = (mag1.int_flyby(N_pt, nlm_o, g_oc[0], h_oc[0], p.r_0, sph_o[0], sph_o[1], sph_o[2],0, P_fly_o[0], P_fly_o[1])[0] + 
                mag1.int_flyby(N_pt, nlm_r, g_res[0], h_res[0], p.r_res, sph_r[0], sph_r[1], sph_r[2], 0, P_fly_r[0], P_fly_r[1])[0] )
    b_fly[0] += (mag1.int_flyby(N_pt, nlm_o, g_oc[n], h_oc[n], p.r_0, sph_o[0], sph_o[1], sph_o[2],0, P_fly_o[0], P_fly_o[1])[0] + 
            mag1.int_flyby(N_pt, nlm_r, g_res[n], h_res[n], p.r_res, sph_r[0], sph_r[1], sph_r[2], 0, P_fly_r[0], P_fly_r[1])[0] )
    
fig, ax = plt.subplots(1,1,figsize=(10,4))


ax.plot(x_r/1e3, b_0_o[0] + b_ind[0], label = r'Ocean induction')
ax.plot(x_r/1e3, b_fly[0] + b_ind[0], color ='k', label = r'Coupled induction')
ax.plot(x_r/1e3, b_sp + b_ind[0], label = r'Superposition')
ax.set_xlim(-200,200)
ax.set_ylabel(r'$B_r$ /nT', fontsize = 14)
ax.set_xlabel(r'$x$ /km', fontsize = 14)
ax.tick_params('both', labelsize = 13)
plt.legend(fontsize = 14)
#plt.savefig('Flyby.pdf', bbox_inches='tight')

sys.close()
print(p.r_res,sph_r[0][250])
print(b_fly[0][250]+b_ind[0][250], b_fly[0][250]-b_0_o[0][250])
#b_int = np.real(-2*p.A_res[1]*217*np.exp(1j*p.phi_res[1,1]))
#print(b_int)
#print(np.abs(b_0_r[500]-b_int)/b_int)
#print(2*np.abs(p.BiBe_res[1])*217*1j*(np.exp(1j*(np.pi/2-p.phi_res[1,1]))))

N= 2
N_pt = 1000
lim_x = [-140e3, 140e3]
lim_y = [p.r_m-100e3, p.r_m+100e3]
lim_z = [0,0]
from CARTGrid import CartGrid
cart_grid = CartGrid(N_pt, N_pt, N_pt, lim_x, lim_y, lim_z)

nlm = p.nlm

r, th, lam = cart_grid.transform_2_sph()
R, LAM = cart_grid.sph_mesh()
R2, LAM2 = cart_grid.sph_mesh_2()
Legendre = LegendrePolynomials(nlm)
P = Legendre.Legendre_array(N_pt, th)
magnetic1 = MagneticField(N_pt, N_pt, nlm, p.omega_m, p.t)

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
B_tot[np.isnan(B_tot)] = 0
norm = plt.Normalize(0, 50)
fig, ax = plt.subplots(figsize=(9,9))
levels = np.arange(25,65,step=5)

#ax.streamplot(cart_grid.X/1e3, cart_grid.Y/1e3, B_ind[0]+fullB_res[0]+fullB_oc[0], B_ind[1]+fullB_res[1]+fullB_oc[1], density=[1], color = 'k')
#CS_r = ax.streamplot(cart_grid.X/1e3, cart_grid.Y/1e3, dipB_res[0]+dipB_oc[0], dipB_oc[1]+dipB_res[1]+B_ind[1],density=[1.5])
CS = ax.streamplot(cart_grid.X/1e3, cart_grid.Y/1e3, fullB_oc[0]+fullB_res[0], fullB_res[1]+fullB_oc[1]+B_ind[1],density=[2], color ='k')
matrix = ax.imshow(B_tot, norm=norm, cmap = 'autumn', extent = [-140,140,lim_y[0]/1e3,lim_y[1]/1e3],origin = 'lower')
cbar = fig.colorbar(matrix, fraction=0.046, pad=0.04)
cbar.set_label(label=r'$B_{tot}$ /nT', size = 16)
cbar.ax.tick_params(labelsize=14)
ax.plot(p.r_0/1e3 * np.cos(np.linspace(0,2*np .pi, num=360)), p.r_0/1e3 * np.sin(np.linspace(0,2*np.pi, num=360)), color = 'k')
ax.plot(p.r_res/1e3 * np.cos(np.linspace(0,2*np.pi)), p.r_c/1e3+p.r_res/1e3 * np.sin(np.linspace(0,2*np.pi)), color = 'k')
ax.plot(p.r_m/1e3 * np.cos(np.linspace(0,2*np .pi, num=360)), p.r_m/1e3 * np.sin(np.linspace(0,2*np.pi, num=360)), color = 'k', linestyle = 'dotted')
ax.set_xlabel(r"$x$ /km", fontsize=16)
ax.set_ylabel(r"$y$ /km", fontsize=16)
ax.tick_params('both', labelsize=15)
ax.set_xlim(-60,60)
ax.set_ylim(1480, 1600)
ax.text(2,1540,'Res', fontsize=17)
ax.text(-7,1492,'Ocean',fontsize=17)
plt.arrow(0, 1540, bind_sc[0], bind_sc[1]/12, width = 0.5,head_width=2, color = 'k', label = 'B_{ind}')
plt.arrow(0,1540, (g_res[1,1,1]+g_res[0,1,1])/6, (h_res[1,1,1]+h_res[0,1,1])/6, width = 0.5,head_width=2, color = 'cyan')
plt.arrow(0,1540, g_oc[0,1,1]/6, h_oc[0,1,1]/6, width = 0.5,head_width=2, color = 'g')
#ax.clabel(CS_r)
#ax.set_yticks([p.r_c/1e3-100,p.r_c/1e3-50,p.r_c/1e3,p.r_c/1e3+50], [-100, -50, 0, 50])
#plt.title(r'Magnetic field lines in the $x-y-$plane')
plt.savefig('BFieldLines_nonSC.pdf')
plt.close()

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
ax.scatter(range(121), R[1], color = 'k', edgecolor = 'k', marker = 'v', s = 25, label = r'Ocean ($n=2$)')
ax.scatter(range(1,2), R_r[0,1], color ='violet', marker = '^', s = 25, label = r'Reservoir ($n=1$)')
ax.scatter(range(121), R_r[1], color = 'violet', edgecolor = 'violet', marker = 'v', s = 25, label = r'Reservoir ($n=2$)')
ax.set_yscale('log')
ax.set_ylim(1e-9,1e5)
ax.set_xlim(0,nlm)
ax.set_xlabel(r'Degree $l$', fontsize = 14)
ax.set_ylabel(r'$R$ /nT$^2$', fontsize = 14)
ax.tick_params('both', labelsize=13)
plt.legend(frameon=False)
plt.savefig('Mauersberger.pdf',bbox_inches='tight')