# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:15:30 2022

@author: Jason
"""

import numpy as np
from LegendrePolynomials import LegendrePolynomials

###CLASS FOR CALCULATING MAGNETIC FIELDS###


class MagneticField(object):
    def __init__(self, nth, nlam, nlm, omega):
        self.nth = nth
        self.nlam = nlam
        self.nlm = nlm+1
        self.omega = omega
        
        
    def calculate_ext(self, nlm_, q, s, C, S):
        self.q = q
        self.s = s
        self.nlm_ = nlm_+1
        self.bfield = np.zeros((self.nth, self.nlam))
        for i in range(self.nth):
            for j in range(self.nlam):
                for l in range(1, self.nlm_):
                    for m in range(0, l+1):
                        self.bfield[i,j] -= l * (self.q[l,m] * C[l,m,i,j] + self.s[l,m] * S[l,m,i,j])
        return self.bfield
    
    def calculate_int(self, nlm, grid, g, h, r, d, th, lam, P_tr, P_tr_dif):
        
        self.nlm = nlm+1
        self.b_r = np.zeros((self.nth, self.nlam))
        self.b_th = np.zeros((self.nth, self.nlam))
        self.b_lam = np.zeros((self.nth, self.nlam))
        
        for l in range(1, self.nlm):
            for m in range(0, l+1):
                self.b_r += (l+1)*(r/d)**(l+2) * P_tr[l][m] * (
                    g[l,m] * np.cos(m*lam) + h[l,m] * np.sin(m*lam))
                self.b_th -= (r/d)**(l+2) * P_tr_dif[l][m] * (
                    g[l,m] * np.cos(m*lam) + h[l,m] * np.sin(m*lam))
                self.b_lam += 1/(np.sin(th)+1e-28) * (r/d)**(l+2) * m * P_tr[l][m] * (
                    g[l,m] * np.sin(m*lam) - h[l,m] * np.cos(m*lam))
        self.bx = (self.b_r * np.sin(th) * np.cos(lam) + self.b_th * np.cos(th) * np.cos(lam) - 
                  self.b_lam * np.sin(lam))
        self.by = (self.b_r * np.sin(th) * np.sin(lam) + self.b_th * np.cos(th) * np.sin(lam) +
                  self.b_lam * np.cos(lam))
        self.bz = self.b_r * np.cos(th) - self.b_th * np.sin(th)
        self.b_r_tr = (self.bx * np.sin(grid.TH) * np.cos(grid.LAM) + self.by * np.sin(grid.TH) * np.sin(grid.LAM) + 
                        self.bz * np.cos(grid.TH))
        return self.b_r_tr
    
    def calculate_int_l(self, l, grid, g, h, r, d, th, lam, P_tr, P_tr_dif):
        
        self.l = l
        self.b_r = np.zeros((self.nth, self.nlam))
        self.b_th = np.zeros((self.nth, self.nlam))
        self.b_lam = np.zeros((self.nth, self.nlam))
        
        for m in range(0, l+1):
            self.b_r += (l+1)*(r/d)**(l+2) * P_tr[l][m] * (
                    g[l,m] * np.cos(m*lam) + h[l,m] * np.sin(m*lam))
            self.b_th -= (r/d)**(l+2) * P_tr_dif[l][m] * (
                    g[l,m] * np.cos(m*lam) + h[l,m] * np.sin(m*lam))
            self.b_lam += 1/(np.sin(th)+1e-28) * (r/d)**(l+2) * m * P_tr[l][m] * (
                    g[l,m] * np.sin(m*lam) - h[l,m] * np.cos(m*lam))
        self.bx = (self.b_r * np.sin(th) * np.cos(lam) + self.b_th * np.cos(th) * np.cos(lam) - 
                  self.b_lam * np.sin(lam))
        self.by = (self.b_r * np.sin(th) * np.sin(lam) + self.b_th * np.cos(th) * np.sin(lam) +
                  self.b_lam * np.cos(lam))
        self.bz = self.b_r * np.cos(th) - self.b_th * np.sin(th)
        self.b_r_tr = (self.bx * np.sin(grid.TH) * np.cos(grid.LAM) + self.by * np.sin(grid.TH) * np.sin(grid.LAM) + 
                        self.bz * np.cos(grid.TH))
        return self.b_r_tr
        
    def b_mean(self, b_r):
        
        self.b_exp = 1/(self.nth*self.nlam)*np.sum(np.sqrt(b_r**2))
        
        return self.b_exp
    def int_test(self, nlm, g, h, C, S):
        self.b_test_int = np.zeros((self.nth, self.nlam))
        for i in range(self.nth):
            for j in range(self.nlam):
                for l in range(1, nlm+1):
                    for m in range(0, l+1):
                        self.b_test_int[i,j] += (l+1) * (g[l,m] * C[l,m,i,j] + h[l,m] * S[l,m,i,j])
                                                        
        return self.b_test_int
    
    def int_test_l(self, g, h, C, S, l):
        self.b_test_int = np.zeros((self.nth, self.nlam))
        for i in range(self.nth):
            for j in range(self.nlam):
                for m in range(0, l+1):
                        self.b_test_int[i,j] += (l+1) * (g[l,m] * C[l,m,i,j] + h[l,m] * S[l,m,i,j])
                                                        
        return self.b_test_int
    
    def mauersberger(self, nlm, g, h):
        self.R = np.zeros(nlm)
        for l in range(1, nlm):
            for m in range(0, l+1):
                self.R[l] += (l+1) * (g[l,m]**2 + h[l,m]**2)
        return self.R
    
    def int_streamline(self, N, nlm, g, h, r, d, th, LAM, P, P_dif):
        
        self.nlm = nlm+1
        self.b_r = np.zeros((N, N))
        self.b_th = np.zeros((N, N))
        self.b_lam = np.zeros((N, N))
        
        for l in range(1, self.nlm):
            for m in range(0, l+1):
                self.b_r += np.real((l+1) * (r/d)**(l+2) * P[l][m] * (
                    g[l,m] * np.cos(m * LAM) + h[l,m] * np.sin(m * LAM)))
                self.b_th -= np.real((r/d)**(l+2) * P_dif[l][m] * (
                    g[l,m] * np.cos(m*LAM) + h[l,m] * np.sin(m*LAM)))
                self.b_lam += np.real(1/(np.sin(th)+1e-28) * (r/d)**(l+2) * m * P[l][m] * (
                    g[l,m] * np.sin(m*LAM) - h[l,m] * np.cos(m*LAM)) )
        self.bx = (self.b_r * np.sin(th) * np.cos(LAM) + self.b_th * np.cos(th) * np.cos(LAM) - 
                  self.b_lam * np.sin(LAM))
        self.by = (self.b_r * np.sin(th) * np.sin(LAM) + self.b_th * np.cos(th) * np.sin(LAM) +
                  self.b_lam * np.cos(LAM))
        self.bz = self.b_r * np.cos(th) - self.b_th * np.sin(th)
        
        self.b_tot = np.sqrt(self.bx**2 + self.by**2 + self.bz**2)
        
        return self.bx, self.by, self.bz, self.b_tot
    
    def int_streamline_l(self, N, nlm, g, h, r, d, th, LAM, P, P_dif, l):
        
        self.nlm = nlm+1
        self.b_r = np.zeros((N, N))
        self.b_th = np.zeros((N, N))
        self.b_lam = np.zeros((N, N))
        
        for m in range(0, l+1):
            self.b_r += np.real((l+1) * (r/d)**(l+2) * P[l][m] * (
                    g[l,m] * np.cos(m * LAM) + h[l,m] * np.sin(m * LAM)))
            self.b_th -= np.real((r/d)**(l+2) * P_dif[l][m] * (
                    g[l,m] * np.cos(m*LAM) + h[l,m] * np.sin(m*LAM)))
            self.b_lam += np.real(1/(np.sin(th)+1e-28) * (r/d)**(l+2) * m * P[l][m] * (
                    g[l,m] * np.sin(m*LAM) - h[l,m] * np.cos(m*LAM)))
        self.bx = (self.b_r * np.sin(th) * np.cos(LAM) + self.b_th * np.cos(th) * np.cos(LAM) - 
                  self.b_lam * np.sin(LAM))
        self.by = (self.b_r * np.sin(th) * np.sin(LAM) + self.b_th * np.cos(th) * np.sin(LAM) +
                  self.b_lam * np.cos(LAM))
        self.bz = self.b_r * np.cos(th) - self.b_th * np.sin(th)
        
        self.b_tot = np.sqrt(self.bx**2 + self.by**2 + self.bz**2)
        
        return self.bx, self.by, self.bz, self.b_tot
    
    def ext_streamline(self, N, nlm_e, q, s, R, d, th, LAM, P, P_dif):
        
        self.nlm_e = nlm_e+1
        self.b_r = np.zeros((N, N))
        self.b_th = np.zeros((N, N))
        self.b_lam = np.zeros((N, N))
        
        for l in range(1, self.nlm):
            for m in range(0, l+1):
                self.b_r -= l * (R/d)**(l-1) * P[l][m] * (
                    q[l,m] * np.cos(m * LAM) + s[l,m] * np.sin(m * LAM))
                self.b_th -= (R/d)**(l-1) * P_dif[l][m] * (
                    q[l,m] * np.cos(m*LAM) + s[l,m] * np.sin(m*LAM))
                self.b_lam += 1/(np.sin(th)+1e-28) * (R/d)**(l-1) * m * P[l][m] * (
                    q[l,m] * np.sin(m*LAM) - s[l,m] * np.cos(m*LAM))
        self.bx = (self.b_r * np.sin(th) * np.cos(LAM) + self.b_th * np.cos(th) * np.cos(LAM) - 
                  self.b_lam * np.sin(LAM))
        self.by = (self.b_r * np.sin(th) * np.sin(LAM) + self.b_th * np.cos(th) * np.sin(LAM) +
                  self.b_lam * np.cos(LAM))
        self.bz = self.b_r * np.cos(th) - self.b_th * np.sin(th)
        
        self.b_tot = np.sqrt(self.bx**2 + self.by**2 + self.bz**2)
        
        return self.bx, self.by, self.bz, self.b_tot
    
    def int_flyby(self, N, nlm, g, h, R, d, th, lam, phi, P, P_dif):
        
        self.nlm=nlm+1
        self.b_r = np.zeros(N)
        self.b_lam = np.zeros(N)
        self.b_th = np.zeros(N)
        for l in range(1, self.nlm):
            for m in range(0, l+1):
                self.b_r += (l+1) * (R/d)**(l+2) * P[l][m] * (
                    g[l,m] * np.cos(m * lam) + h[l,m] * np.sin(m * lam))
                self.b_th -= (R/d)**(l+2) * P_dif[l][m] * (
                    g[l,m] * np.cos(m * lam) + h[l,m] * np.sin(m * lam))
                self.b_lam += 1/(np.sin(th)) * (R/d)**(l+2) * P[l][m] * (
                    g[l,m] * np.sin(m * lam) - h[l,m] * np.cos(m * lam))
        self.bx = (self.b_r * np.sin(th) * np.cos(lam) + self.b_th * np.cos(th) * np.cos(lam) - 
                  self.b_lam * np.sin(lam))
        self.by = (self.b_r * np.sin(th) * np.sin(lam) + self.b_th * np.cos(th) * np.sin(lam) +
                  self.b_lam * np.cos(lam))
        self.bz = self.b_r * np.cos(th) - self.b_th * np.sin(th)
                
        return self.bx, self.b_lam, self.b_th
    
    def B_ind(self, B0x, B0y, phix, phiy, phi, t):
        
        B_indx = np.real(B0x * np.exp(1j*(self.omega*t+phix+phi)))
        B_indy = np.real(B0y * np.exp(1j*(self.omega*t+phiy+phi)))
        
        return B_indx, B_indy
                    