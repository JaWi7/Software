# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:20:17 2022

"""

import numpy as np

class Integrator(object):
    """
    
    The Integrator class performs the integration for external Gauss coefficients
    (see equation (9) in Winkenstern et al., 2023). For that, a Riemann sum is used.
    
    """
    
    def __init__(self, nlm, dth, dlam):
        """
        
        Parameters
        ----------
        nlm : int
            Maximum number of degrees/orders 
        dth : float
            Gridsteps along latitudinal direction
        dlam : float
            Gridsteps along longitudinal direction

        Returns
        -------
        None.

        """
        self.nlm = nlm+1
        self.dth = dth
        self.dlam = dlam
        
    def integrate(self, grid, b_r):
        """

        Parameters
        ----------
        grid : class
            The grid class that defines the surface over which the integration is 
            performed.
        b_r : array
            The radial component of the inducing field across the surface

        Returns
        -------
        result_q : array
            External Gauss coefficient q[l,m] of degree l and order m
        result_s : array
            External Gauss coefficient s[l,m] of degree l and order m

        """
        
        result_q = np.zeros((self.nlm,self.nlm))
        result_s = np.zeros((self.nlm,self.nlm))
        
        for l in range(1, self.nlm):
            for m in range(l+1):
                integrand_q = grid.int_const_q[l][m] * b_r * self.dth * self.dlam
                integrand_s = grid.int_const_s[l][m] * b_r * self.dth * self.dlam
        
                result_q[l,m] = np.sum(integrand_q)
                result_s[l,m] = np.sum(integrand_s)
        
        return result_q, result_s