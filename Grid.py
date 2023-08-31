# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:19:24 2022
"""

import numpy as np


class Grid(object):
    """
    
    This class creates the grid used for induction calculations
    Input requires number of grid points along latitude, longitude, and limits
    Also calculates the transformation between ocean and reservoir-centered coordinates 
    (CURRENTLY HARD CODED TO X-AXIS!)
    
    """
    def __init__(self, nth, nlam, lim_th, lim_lam):
        """
        
        Parameters
        ----------
        nth : int
            Number of grid points along latitude
        nlam : int
            Number of grid points along longitude
        lim_th : float
            Latitudinal coverage in rad
        lim_lam : float
            Longitudinal coverage in rad
            
        """
        self.nth = nth
        self.nlam = nlam
        self.lim_th = lim_th
        self.lim_lam = lim_lam
        
        self.th = np.linspace(lim_th[0], lim_th[1], nth, endpoint = False)
        self.lam = np.linspace(lim_lam[0], lim_lam[1], nlam, endpoint = False)
    
        self.nlm = None
        self.int_const = None
        
        self.LAM, self.TH = np.meshgrid(self.lam, self.th)
        
        
    def calculate_int_constants(self, nlm, P):
        """
        
        Parameters
        ----------
        nlm : int
            Maximum number of degrees/orders.
        P : array
            Associated Legendre Polyomials.

        Returns
        -------
        array
            Integrand (excl bfield) for external Gauss coefficients q_l^m
        array
            Integrand (excl bfield) for external Gauss coefficients s_l^m.

        """
        self.nlm = nlm+1
        self.int_const_q = np.zeros((self.nlm, self.nlm, self.nth, self.nlam))
        self.int_const_s = np.zeros((self.nlm, self.nlm, self.nth, self.nlam))
        
        for l in range(1,nlm+1):
            for m in range(l+1):
                self.int_const_q[l,m] = -(2*l + 1)/(4* np.pi * l) * np.sin(self.TH) * P[l,m] * np.cos(m*self.LAM)
                self.int_const_s[l,m] = -(2*l + 1)/(4* np.pi * l) * np.sin(self.TH) * P[l,m] * np.sin(m*self.LAM)
        return self.int_const_q, self.int_const_s
    
    def transform(self, r_c, r_0, r_res, th_c, lam_c):
        """
        Calculates the spherical coordinates of the grid spun across the surface of the
        reservoir/ocean in the coordinate system with center in Europa/reservoir

        Parameters
        ----------
        r_c : float
            Radial distance from reservoir-Europa center
        r_0 : float
            Outer radius ocean (i.e., ocean surface)
        r_res : float
            Reservoir radius
        th_c : float
            Latitude of reservoir position (in Europa IAU)
        lam_c : float
            Longitude of reservoir position (in Europa IAU)

        Returns
        -------
        float
            Radial component of reservoir surface grid in Europa-centered coordinates.
        float
            Radial component of ocean surface grid in reservoir-centered coordinates.
        float
            Longitudinal component of reservoir surface grid in Europa-centered coordinates.
        float
            Longitudinal component of ocean surface grid in reservoir-centered coordinates.
        float
            Latitudinal component of reservoir surface grid in Europa-centered coordinates.
        float
            Latitudinal component of ocean surface grid in reservoir-centered coordinates.

        """
        self.r_c = r_c
        self.r_0 = r_0
        self.r_res = r_res
        self.th_c = th_c
        self.lam_c = lam_c
        
        self.x = r_c * np.sin(th_c) * np.cos(lam_c) + r_res * np.sin(self.TH) * np.cos(self.LAM)
        self.y = r_c * np.sin(th_c) * np.sin(lam_c) + r_res * np.sin(self.TH) * np.sin(self.LAM)
        self.z = r_c * np.cos(th_c)  + r_res * np.cos(self.TH)

        self.tx = -r_c * np.sin(th_c) * np.cos(lam_c) + r_0 * np.sin(self.TH) * np.cos(self.LAM)
        self.ty = -r_c * np.sin(th_c) * np.sin(lam_c) + r_0 * np.sin(self.TH) * np.sin(self.LAM)
        self.tz = -r_c * np.cos(th_c) + r_0 * np.cos(self.TH)

    	################SPHERICAL COMPONENTS################

        self.d_oc = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.lam_oc = np.arctan2(self.y, self.x)
        self.th_oc = np.arctan2(np.sqrt(self.y**2 + self.x**2), self.z)

        self.d_res = np.sqrt(self.tx**2 + self.ty**2 + self.tz**2)
        self.lam_res = np.arctan2(self.ty, self.tx)
        self.th_res = np.arctan2(np.sqrt(self.ty**2 + self.tx**2), self.tz)
        
        return self.d_oc, self.d_res, self.lam_oc, self.lam_res, self.th_oc, self.th_res
        
    def moonsurface(self, r_c, r_0, r_res, r_m, th_c, lam_c):
        """
        Calculates the spherical coordinates of the grid spun across the natural
        satellite's surface in coordinates with center in Europa and reservoir

        Parameters
        ----------
        r_c : float
            Radial distance from reservoir-Europa center
        r_0 : float
            Outer radius ocean (i.e., ocean surface)
        r_res : float
            Reservoir radius
        th_c : float
            Latitude of reservoir position (in Europa IAU)
        lam_c : float
            Longitude of reservoir position (in Europa IAU)
            
        Returns
        -------
        float
            Radial component of moon surface grid in Europa-centered coordinates.
        float
            Radial component of moon surface grid in reservoir-centered coordinates.
        float
            Longitudinal component of moon surface grid in Europa-centered coordinates.
        float
            Longitudinal component of moon surface grid in reservoir-centered coordinates.
        float
            Latitudinal component of moon surface grid in Europa-centered coordinates.
        float
            Latitudinal component of moon surface grid in reservoir-centered coordinates.

        """
        self.r_m = r_m
        self.r_0 = r_0
        self.r_res = r_res
        self.r_c = r_c
        
        self.x_m_o = self.r_m * np.sin(self.TH) * np.cos(self.LAM)
        self.y_m_o = self.r_m * np.sin(self.TH) * np.sin(self.LAM)
        self.z_m_o = self.r_m * np.cos(self.TH)
        
        self.x_m_r = -r_c * np.sin(th_c) * np.cos(lam_c) + self.r_m * np.sin(self.TH) * np.cos(self.LAM)
        self.y_m_r = -r_c * np.sin(th_c) * np.sin(lam_c) + self.r_m * np.sin(self.TH) * np.sin(self.LAM)
        self.z_m_r = -r_c * np.cos(th_c) + self.r_m * np.cos(self.TH)
        
        ################SPHERICAL COMPONENTS##################
        
        self.d_m_o = np.sqrt(self.x_m_o**2 + self.y_m_o**2 + self.z_m_o**2)
        self.lam_m_o = np.arctan2(self.y_m_o, self.x_m_o)
        self.th_m_o = np.arctan2(np.sqrt(self.y_m_o**2 + self.x_m_o**2), self.z_m_o)

        self.d_m_r = np.sqrt(self.x_m_r**2 + self.y_m_r**2 + self.z_m_r**2)
        self.lam_m_r = np.arctan2(self.y_m_r, self.x_m_r)
        self.th_m_r = np.arctan2(np.sqrt(self.y_m_r**2 + self.x_m_r**2), self.z_m_r)
        
        return self.d_m_o, self.d_m_r, self.lam_m_o, self.lam_m_r, self.th_m_o, self.th_m_r
        