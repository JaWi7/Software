# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:19:24 2022

This class creates the grid used for induction calculations
Input requires number of grid points along latitude, longitude, and limits
Also calculates the transformation between ocean and reservoir-centered coordinates 
(CURRENTLY HARD CODED TO X-AXIS!)
"""

import numpy as np


class Grid(object):
    def __init__(self, nth, nlam, lim_th, lim_lam):
        self.nth = nth
        self.nlam = nlam
        self.lim_th = lim_th
        self.lim_lam = lim_lam
        
        self.th = np.linspace(lim_th[0], lim_th[1], nth, endpoint = False)
        self.lam = np.linspace(lim_lam[0], lim_lam[1], nlam, endpoint = False)
    
        self.nlm = None
        self.int_const = None
        
        self.LAM, self.TH = np.meshgrid(self.lam, self.th)
        
        
    def calculate_int_constants(self, nlm, C, S):
        self.nlm = nlm+1
        self.C = C
        self.S = S
        self.int_const_q = np.zeros((self.nlm, self.nlm, self.nth, self.nlam))
        self.int_const_s = np.zeros((self.nlm, self.nlm, self.nth, self.nlam))
        
        for i in range(self.nth):
            for j in range(self.nlam):
                for l in range(1, nlm+1):
                    for m in range(0, l+1):
                        self.int_const_q[l,m,i,j] = -(2*l + 1)/(4* np.pi * l) * np.sin(self.th[i]) * self.C[l,m,i,j]
                        self.int_const_s[l,m,i,j] = -(2*l + 1)/(4* np.pi * l) * np.sin(self.th[i]) * self.S[l,m,i,j]
        return self.int_const_q, self.int_const_s
    
    def transform(self, r_c, r_0, r_res, th_c, lam_c):
        self.r_c = r_c
        self.r_0 = r_0
        self.r_res = r_res
        self.th_c = th_c
        self.lam_c = lam_c
        
        self.x = r_c + r_res * np.sin(self.TH) * np.cos(self.LAM)
        self.y = r_res * np.sin(self.TH) * np.sin(self.LAM)
        self.z = r_res * np.cos(self.TH)

        self.tx = -r_c + r_0 * np.sin(self.TH) * np.cos(self.LAM)
        self.ty = r_0 * np.sin(self.TH) * np.sin(self.LAM)
        self.tz = r_0 * np.cos(self.TH)

    	################SPHERICAL COMPONENTS################

        self.d_oc = np.sqrt(self.r_res**2 + self.r_c**2 + 2*self.r_res*self.r_c * np.sin(self.TH) * np.sin(self.LAM))
        self.lam_oc = np.arctan2(self.y, self.x)
        self.th_oc = np.arctan2(np.sqrt(self.y**2 + self.x**2), self.z)

        self.d_res = np.sqrt(self.r_0**2 + self.r_c**2 - 2*self.r_0*self.r_c * np.sin(self.TH) * np.sin(self.LAM))
        self.lam_res = np.arctan2(self.ty, self.tx)
        self.th_res = np.arctan2(np.sqrt(self.ty**2 + self.tx**2), self.tz)
        
        return self.d_oc, self.d_res, self.lam_oc, self.lam_res, self.th_oc, self.th_res
        
    def moonsurface(self, r_c, r_0, r_res, r_m):
        self.r_m = r_m
        self.r_0 = r_0
        self.r_res = r_res
        self.r_c = r_c
        
        self.x_m_o = self.r_m * np.sin(self.TH) * np.cos(self.LAM)
        self.y_m_o = self.r_m * np.sin(self.TH) * np.sin(self.LAM)
        self.z_m_o = self.r_m * np.cos(self.TH)
        
        self.x_m_r = -self.r_c + self.r_m * np.sin(self.TH) * np.cos(self.LAM)
        self.y_m_r = self.r_m * np.sin(self.TH) * np.sin(self.LAM)
        self.z_m_r = self.r_m * np.cos(self.TH)
        
        ################SPHERICAL COMPONENTS##################
        
        self.d_m_o = np.sqrt(self.x_m_o**2 + self.y_m_o**2 + self.z_m_o**2)
        self.lam_m_o = np.arctan2(self.y_m_o, self.x_m_o)
        self.th_m_o = np.arctan2(np.sqrt(self.y_m_o**2 + self.x_m_o**2), self.z_m_o)

        self.d_m_r = np.sqrt(self.x_m_r**2 + self.y_m_r**2 + self.z_m_r**2)
        self.lam_m_r = np.arctan2(self.y_m_r, self.x_m_r)
        self.th_m_r = np.arctan2(np.sqrt(self.y_m_r**2 + self.x_m_r**2), self.z_m_r)
        
        return self.d_m_o, self.d_m_r, self.lam_m_o, self.lam_m_r, self.th_m_o, self.th_m_r
        