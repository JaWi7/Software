# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 18:29:52 2022

@author: Jason

@Use this to calculate Cartesian Grids for Fieldline Plots
"""

import numpy as np
import Parameters as p

class CartGrid(object):
    def __init__(self, NX, NY, NZ, lim_x, lim_y, lim_z):
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.lim_x = lim_x
        self.lim_y = lim_y
        self.lim_z = lim_z
        
        self.x = np.linspace(lim_x[0], lim_x[1], self.NX, endpoint=False)
        self.y = np.linspace(lim_y[0], lim_y[1], self.NY, endpoint=False)
        self.z = np.linspace(lim_z[0], lim_z[1], self.NZ, endpoint=False)
        
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def transform_2_sph(self):
        
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.th = np.arctan2(np.sqrt(self.x**2+self.y**2),self.z)
        self.lam = np.arctan2(self.y, self.x)
        
        return self.r, self.th, self.lam
    
    def sph_mesh(self):
        
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.LAM = np.arctan2(self.Y, self.X)
        
        return self.R, self.LAM
    
    def sph_mesh_2(self):
       
        self.y_ = self.y - (p.r_c)
       
        self.X_, self.Y_ = np.meshgrid(self.x, self.y_)
        self.R2 = np.sqrt(self.X_**2+self.Y_**2)
        self.LAM2 = np.arctan2(self.Y_, self.X_)
        
        return self.R2, self.LAM2