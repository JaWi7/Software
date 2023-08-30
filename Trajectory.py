# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:06:08 2022

Creates the coordinates of any trajectory along which the magnetic fields
are to be calculated.

"""
import numpy as np
from LegendrePolynomials import LegendrePolynomials 

class Trajectory(object):
    def __init__(self, n, x_a, x_b, y_a, y_b, z_a, z_b):
        self.n = n
        self.x_a = x_a
        self.x_b = x_b
        self.y_a = y_a
        self.y_b = y_b
        self.z_a = z_a
        self.z_b = z_b
        
        self.x = np.linspace(x_a, x_b, n, endpoint = False)
        self.y = np.linspace(y_a, y_b, n, endpoint = False)
        self.z = np.linspace(z_a, z_b, n, endpoint = False)
    
    def cart_coords(self):
        
        return self.x, self.y, self.z
        
    def sph_coords(self):
        
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.th = np.arctan2(np.sqrt(self.x**2+self.y**2),self.z)
        self.lam = np.arctan2(self.y, self.x)
        
        return self.r, self.th, self.lam

