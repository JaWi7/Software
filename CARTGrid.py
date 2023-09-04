# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 18:29:52 2022

While Grid.py yields the spherical grid used to calculate induced and inducing
fields at the surface of reservoir and ocean, CARTGrid.py offers a Cartesian grid,
which in this work was used to visualize the Bx,By-components in the equatorial
plane around the reservoir.
 
"""

import numpy as np
import Parameters as p

class CartGrid(object):
    def __init__(self, NX, NY, lim_x, lim_y, z):
        """
        Defines the grid in the xy-plane. 
        

        Parameters
        ----------
        NX : int
            Number of grid points along x-direction.
        NY : TYPE
            Number of grid points along y-direction.
        lim_x : list
            Initial and final points of the grid in x-direction.
        lim_y : list
            Initial and final points of the grid in y-direction.
        z : float
            z coordinate of the xy-plane.

        Returns
        -------
        None.

        """
        self.NX = NX
        self.NY = NY
        self.lim_x = lim_x
        self.lim_y = lim_y

        
        self.x = np.linspace(lim_x[0], lim_x[1], self.NX, endpoint=False)
        self.y = np.linspace(lim_y[0], lim_y[1], self.NY, endpoint=False)
        self.z = z
        
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def transform_2_sph(self):
        """
        Defines the spherical coordinates of the xy-plane in Europa-centered
        coordinates.

        Returns
        -------
        array
            Radial component in Europa-centered coordinates.
        array
            Latitudinal component in Europa-centered coordinates.
        array
            Longitudinal component in Europa-centered coordinates.

        """
        
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.th = np.arctan2(np.sqrt(self.x**2+self.y**2),self.z)
        self.lam = np.arctan2(self.y, self.x)
        
        return self.r, self.th, self.lam
    
    def sph_mesh(self):
        """
        Defines the spherical grid in Europa-centered coordinates. Used for
        calculating the magnetic field in the xy-plane.

        Returns
        -------
        array
            Radial component in Europa-centered spherical coordinates.
        array
            Longitudinal component in Europa-centered spherical coordinates.

        """
        
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.LAM = np.arctan2(self.Y, self.X)
        
        return self.R, self.LAM
    
    def sph_mesh_2(self):
        """
        Defines the spherical grid in reservoir-centered coordinates. Used for
        calculating the magnetic field in the xy-plane.

        Returns
        -------
        array
            Radial component in reservoir-centered spherical coordinates.
        array
            Longitudinal component in reservoir-centered spherical coordinates.

        """
       
        self.x_ = self.x - (p.r_c)
       
        self.X_, self.Y_ = np.meshgrid(self.x_, self.y)
        self.R2 = np.sqrt(self.X_**2+self.Y_**2)
        self.LAM2 = np.arctan2(self.Y_, self.X_)
        
        return self.R2, self.LAM2
