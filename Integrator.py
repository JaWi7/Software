# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:20:17 2022

This class defines the integration used to obtain external Gauss coefficients
q_l^m, s_l^m of degree l and order m

"""

import numpy as np

class Integrator(object):
    def __init__(self, nlm, dth, dlam):
        self.nlm = nlm+1
        self.dth = dth
        self.dlam = dlam
        
    def integrate(self, grid, b_r):
        
        result_q = np.zeros((self.nlm,self.nlm))
        result_s = np.zeros((self.nlm,self.nlm))
        
        for l in range(1, self.nlm):
            for m in range(l+1):
                integrand_q = grid.int_const_q[l][m] * b_r * self.dth * self.dlam
                integrand_s = grid.int_const_s[l][m] * b_r * self.dth * self.dlam
        
                result_q[l,m] = np.sum(integrand_q)
                result_s[l,m] = np.sum(integrand_s)
        
        return result_q, result_s