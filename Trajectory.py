# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:06:08 2022

"""

import numpy as np
from LegendrePolynomials import LegendrePolynomials 

class Trajectory(object):
    """
    
    Creates the coordinates of any trajectory along which the magnetic fields
    are to be calculated.
    
    """
    def __init__(self, n, x_a, x_b, y_a, y_b, z_a, z_b):
        """
        
        Parameters
        ----------
        n : int
            Number of points along the trajectory.
        x_a : float
            x-component of the initial point of the trajectory (in m).
        x_b : float
            x-component of the final point of the trajectory (in m).
        y_a : float
            y-component of the initial point of the trajectory (in m).
        y_b : float
            y-component of the final point of the trajectory (in m).
        z_a : float
            z-component of the initial point of the trajectory (in m).
        z_b : float
            z-component of the final point of the trajectory (in m).

        Returns
        -------
        None.

        """
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
        """
        Returns the cartesian coordinates of the trajectory

        Returns
        -------
        array
            x coordinates of the trajectory (in m).
        array
            y coordinates of the trajectory (in m).
        array
            z coordinates of the trajectory (in m).

        """
        
        return self.x, self.y, self.z
        
    def sph_coords(self):
        """
        Returns the spherical coordinates of the trajectory 

        Returns
        -------
        array
            r coordinates of the trajectory (in m).
        array
            latitudinal coordinates of the trajectory (in rad).
        array
            longitudinal coordinates of the trajectory (in rad).

        """
        
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.th = np.arctan2(np.sqrt(self.x**2+self.y**2),self.z)
        self.lam = np.arctan2(self.y, self.x)
        
        return self.r, self.th, self.lam

