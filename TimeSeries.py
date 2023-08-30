# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:36:08 2022

This script solves the mutual induction coupling along a full synodic period
These time series can then be plotted and the maximum deviation between 
the measurements of two magnetometers deployed on Europa's surface can be found

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


nlm = p.nlm #maximum degree/order l,m considered
nlm_o = 4
nlm_r = 4
t = np.linspace(0,p.T,num=73,endpoint=True)

grid1 = Grid(p.NTH, p.NLAM, (0, np.pi), (0, 2*np.pi)) #Grid Setup!!!
Legendre = LegendrePolynomials(nlm) #Legendre Class!!!
C, S = Legendre.baseline(grid1)
integ = Integrator(nlm, p.dth, p.dlam)
grid1.calculate_int_constants(nlm, C, S)
grid1.transform(p.r_c, p.r_0, p.r_res, p.th_c, p.lam_c) #Calls transformed coordinates
grid1.moonsurface(p.r_c, p.r_0, p.r_res, p.r_m)
P_tr_o, P_tr_o_dif = Legendre.transformed_Legendre(grid1, nlm_o, grid1.th_oc, grid1.lam_oc) #res surface from oc frame
P_tr_r, P_tr_r_dif = Legendre.transformed_Legendre(grid1, nlm_r, grid1.th_res, grid1.lam_res) #oc surface from res frame

P_tr_m_o, P_tr_m_o_dif = Legendre.transformed_Legendre(grid1, nlm, grid1.th_m_o, grid1.lam_m_o)
P_tr_m_r, P_tr_m_r_dif = Legendre.transformed_Legendre(grid1, nlm, grid1.th_m_r, grid1.lam_m_r)

###Set up baseline iteration parameters###

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

N = 2    #number of iterations
K = 2 #0 superconduct, 1 finite

k_2_far_r = np.zeros((K+1,N,len(p.sigma_res),len(t)))
k_2_near_r = np.zeros((K+1,N,len(p.sigma_res),len(t)))
k_2_far_o = np.zeros((K+1,N,len(p.sigma_res),len(t)))
k_2_near_o = np.zeros((K+1,N,len(p.sigma_res),len(t)))

B_sph = np.zeros((len(p.sigma_res),len(t)))
B_ind = np.zeros(len(t))

for s in range(len(p.sigma_res)):
    B_ext_res = np.zeros((len(t),p.NTH, p.NLAM))
    B_ext_oc = np.zeros((len(t),p.NTH, p.NLAM))
    print('------------------------------------------')
    print('-----Conductivity:', p.sigma_res[s], 'S/m-----')
    for i in range(len(t)):
        if int(p.omega_m*t[i]*180/np.pi)%90 == 0:
            print('Timestep:', p.omega_m*t[i]*180/np.pi)
        g_oc = np.zeros((N,nlm+1,nlm+1))
        h_oc = np.zeros((N,nlm+1,nlm+1))
    
        g_res = np.zeros((N,nlm+1,nlm+1))
        h_res = np.zeros((N,nlm+1,nlm+1))
    
        q_oc = np.zeros((nlm+1,nlm+1))
        s_oc = np.zeros((nlm+1,nlm+1))
    
        q_res = np.zeros((nlm+1,nlm+1))
        s_res = np.zeros((nlm+1,nlm+1))
        
        
        
        mag1 = MagneticField(p.NTH, p.NLAM, nlm, p.omega_m) #Magnetic Field Class!!!
        bindo = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, p.phi[1], t[i])
        bind_sc = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, 0, t[i])
        bindr = mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, p.phi_res[1,s], t[i])
        
        B_ind[i] = -bind_sc[0]
        qjo = -bindo[0]
        sjo = -bindo[1]
        
        qjr = -bindr[0]
        sjr = -bindr[1]
        q_sc = -bind_sc[0]
        s_sc = -bind_sc[1]
        
        g_oc[0,1,1] = np.abs(p.BiBe[1])*qjo
        h_oc[0,1,1] = np.abs(p.BiBe[1])*sjo
    
        g_res[0,1,1] = np.abs(p.BiBe_res[1,s])*qjr
        h_res[0,1,1] = np.abs(p.BiBe_res[1,s])*sjr
        
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
        
    
        k_2_far_r[1,0,s,i] = 1
        k_2_near_r[1,0,s,i] = 1
        k_2_far_o[1,0,s,i] = 1
        k_2_near_o[1,0,s,i] = 1
    
        B_ext_r += mag1.calculate_int(1, grid1, g_oc[0], h_oc[0], p.r_0, grid1.d_oc, grid1.th_oc, grid1.lam_oc, P_tr_o, P_tr_o_dif)
        b_ext_r[0] = mag1.calculate_int(1, grid1, g_oc[0], h_oc[0], p.r_0, grid1.d_oc, grid1.th_oc, grid1.lam_oc, P_tr_o, P_tr_o_dif)
        B_int_r += mag1.int_test(1,g_res[0], h_res[0], C, S)
    
        B_ext_o += mag1.calculate_int(1, grid1, g_res[0], h_res[0], p.r_res, grid1.d_res, grid1.th_res, grid1.lam_res, P_tr_r, P_tr_r_dif)
        b_ext_o[0] = mag1.calculate_int(1, grid1, g_res[0], h_res[0], p.r_res, grid1.d_res, grid1.th_res, grid1.lam_res, P_tr_r, P_tr_r_dif)
        B_int_o += mag1.int_test(1,g_oc[0], h_oc[0], C, S)
    
        b_ind_0_r = mag1.int_test(1, g_res[0], h_res[0], C, S)
        b_ind_0_o = mag1.int_test(1, g_oc[0], h_oc[0], C, S)
    
        k_2_far_r[2,0,s,i]  = B_int_r[90,90] + B_ext_r[90,90]
        k_2_near_r[2,0,s,i] = B_int_r[90,270] + B_ext_r[90,270]
    
        k_2_near_o[2,0,s,i]  = B_int_o[90,90] + B_ext_o[90,90]
        k_2_far_o[2,0,s,i] = B_int_o[90,270] + B_ext_o[90,270]
        start_time = time.time()
    
    
        z = 0
        count = 0
        phi_o = np.zeros(nlm+1)
        phi_r = np.zeros(nlm+1)
        for n in range(1,N):
            ###finite###
            if n == 1:
                for l in range(1,nlm+1):
                    phi_r[l] = p.phi[l] + p.phi_res[1,s]
                    phi_o[l] = p.phi_res[l,s] + p.phi[1]
                    q_J_oc = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_o[count+l], t[i])[0]
                    s_J_oc = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_o[count+l], t[i])[1]
                    q_J_res = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_r[count+l], t[i])[0]
                    s_J_res = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_r[count+l], t[i])[1]
                    GG = GetGauss(q_J_oc, s_J_oc, q_J_res, s_J_res, n, [1,l])
                    if l == 1:
                        #g_oc[n,l] += GG[0][0][l]
                        #h_oc[n,l] += GG[1][0][l]
                        g_res[n,l] += GG[0][1][l]
                        h_res[n,l] += GG[1][1][l]
                        q_oc += GG[2]
                        s_oc += GG[3]
                        q_res += GG[4]
                        s_res += GG[5]
                        b_tr_o += GG[6]
                        b_tr_r += GG[7]
                    elif l > 1:
                        #g_oc[n,l] += GG[0][0][l]
                        #h_oc[n,l] += GG[1][0][l]
                        g_res[n,l] += GG[0][1][l]
                        h_res[n,l] += GG[1][1][l]
    
            # else:
            #     L = np.zeros((n,nlm**n+1))
            #     count += nlm**(n-1) 
            #     phi_r = np.append(phi_r,np.zeros(nlm**n))
            #     phi_o = np.append(phi_o,np.zeros(nlm**n))
            #     print(len(phi_r))
            #     for l in range(1,nlm**n+1):
            #         L[0,l] = int((l-1)/(nlm**(n-1)))+1
            #         w = (l-1)%nlm
            #         v = (l-1)/nlm
            #         for n_ in range(0,n):
            #             if 0 < n_ < n-1:
            #                 L[n_,l] = int((l-1)/(nlm**(n - (n_+1))))+1
            #                 for j in range(1, nlm+1):
            #                     L[L%nlm == j] = j
            #                     L[L%nlm == 0] = nlm
            #             elif n_ == n-1:
            #                 L[n_,l] = w+1
            #                 if (n%2) == 0:
            #                     phi_r[count+l] = phi_r[count-(nlm)**(n-1)+int(v)+1] + p.phi_res[w+1]
            #                     phi_o[count+l] = phi_o[count-(nlm)**(n-1)+int(v)+1] + p.phi[w+1]
            #                 if (n%2) == 1:
            #                     phi_r[count+l] = phi_r[count-(nlm)**(n-1)+int(v)+1] + p.phi[w+1]
            #                     phi_o[count+l] = phi_o[count-(nlm)**(n-1)+int(v)+1] + p.phi_res[w+1]
            #                 q_J_oc = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_o[count+l], t[i])[0]
            #                 s_J_oc = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_o[count+l], t[i])[1]
            #                 q_J_res = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_r[count+l], t[i])[0]
            #                 s_J_res = -mag1.B_ind(p.B_0[0], p.B_0[1], 0, -np.pi/2, phi_r[count+l], t[i])[1]
            #                 GG = GetGauss(q_J_oc, s_J_oc, q_J_res, s_J_res, n, L[:,l])
            #                 # q_oc[n-1,w+1] += GG[2]
            #                 # s_oc[n-1,w+1] += GG[3]
            #                 # q_res[n-1,w+1] += GG[4]
            #                 # s_res[n-1,w+1] += GG[5]
            #         if w <= nlm_r:
            #             print('Adding to both.')
            #             g_oc[n,w+1] += GG[0][0][w+1]
            #             h_oc[n,w+1] += GG[1][0][w+1]
            #             g_res[n,w+1] += GG[0][1][w+1]
            #             h_res[n,w+1] += GG[1][1][w+1]
            #         elif w > nlm_r and w <= nlm_o:
            #             print('Adding to O only.')
            #             g_oc[n,w+1] += GG[0][0][w+1]
            #             h_oc[n,w+1] += GG[1][0][w+1]
                        
    
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
    
            # k_2_far_r[1,n,s,i] = B_int_r[90,90]/b_ind_0_r[90,90] * 2*np.abs(p.BiBe_res[1,s])
            # k_2_near_r[1,n,s,i] = (B_ext_r[90, 270] + B_int_r[90,270])/(b_ext_r[0][90,270] + b_ind_0_r[90,270])
            
            k_2_far_r[2,n,s,i] = B_int_r[90,0] + B_ext_r[90,0]
            # k_2_near_r[2,n,s,i] = B_int_r[90,270] + B_ext_r[90,270]
            
            # k_2_far_o[1,n,s,i] = (B_ext_o[90, 270] + B_int_o[90,270])/(b_ext_o[0][90,270] + b_ind_0_o[90,270])
            # k_2_near_o[1,n,s,i] = (B_ext_o[90, 90] + B_int_o[90,90])/(b_ext_o[0][90,90] + b_ind_0_o[90,90])
            
            # k_2_far_o[2,n,s,i] = B_int_o[90,270] + B_ext_o[90,270]
            # k_2_near_o[2,n,s,i] = B_int_o[90,90] + B_ext_o[90,90]
            
            B_sph[s,i] = B_ext_r[90,0]
        
        B_ext_res[i] = (mag1.calculate_int(nlm_r, grid1, g_res[0]+g_res[1], h_res[0]+h_res[1], p.r_res, grid1.d_m_r, grid1.th_m_r, 
                                                  grid1.lam_m_r, P_tr_m_r, P_tr_m_r_dif))
        B_ext_oc[i] = (mag1.calculate_int(nlm_o, grid1, g_oc[0], h_oc[0], p.r_0, grid1.d_m_o, grid1.th_m_o, 
                                    grid1.lam_m_o, P_tr_m_o, P_tr_m_o_dif))
###Seed starting parameters###

    Bre_on_c = np.zeros(len(t))
    Bre_off_c = np.zeros(len(t))
    Boc_on_c = np.zeros(len(t))
    Boc_off_c = np.zeros(len(t))
    for j in range(len(t)):
        Bre_on_c[j] = B_ext_res[j][90,0]
        Bre_off_c[j] = B_ext_res[j][90,359]
        Boc_on_c[j] = B_ext_oc[j][90,0]
        Boc_off_c[j] = B_ext_oc[j][90,359]
    
    fig, ax = plt.subplots(figsize=(9,7))
    ax.plot(t/3600, Bre_on_c+Boc_on_c+B_ind, color ='k')
    ax.plot(t/3600, Bre_off_c+Boc_off_c+B_ind, color='k',linestyle = 'dashdot')
    ax2 = ax.twinx()
    ax2.spines['right'].set_color('red')
    ax2.plot(t/3600,Boc_on_c-Boc_off_c, color ='r')
    ax.set_xlim(0,p.T/3600)
    ax.set_xlabel(r'Time $t$ /h',fontsize=14)
    ax.set_ylabel(r'$B_r$ /nT',fontsize=14)
    #ax.set_ylim(-65,65)
    ax.tick_params(axis='both', labelsize = 13)
    ax2.set_ylim(-10,10)
    ax2.tick_params(axis='y', colors='red', labelsize=13)
    ax2.set_ylabel(r'$\Delta B_r$ /nT', color ='r',fontsize=14)
    #plt.savefig('Timeseries_oc.pdf',bbox_inches = 'tight')
    
    print('Maximum difference: ' ,np.max(Bre_on_c-Bre_off_c))
    print('At time:', t[np.argmax(Bre_on_c-Bre_off_c)]/3600, 'h')


    