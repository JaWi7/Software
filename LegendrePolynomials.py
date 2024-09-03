# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:14:15 2022

"""


import numpy as np 
from scipy import special
import math


class LegendrePolynomials(object):
    def __init__(self, grid, nlm):
        """
        This class defines the Schmidt quasi-normalized assoociated Legendre 
        polynomials. These are calculated using SciPy implemented functions.
        
        NOTE: This calcualtion is only suitable for calculations with nlm < 89.
        This is due to the use of the factorial function in the normalizing values,
        which gives zero for large values in the denominator.
        Calculations with nlm => 89 work using a recusion formula, where the
        factorials are cancelled out, however this is will greately decrease
        the peformance of the script with a greatly increased runtime.
        Outside of numerical accuracy testing, nlm < 89 will suffice.
        
        Parameters
        ----------
        grid : class
            Corresponding grid class that defines the number of grid points
        nlm : int
            Maximum number of degrees l and orders m.

        Returns
        -------
        None.

        """
        self.nth = grid.nth
        self.nlam = grid.nlam
        self.th = grid.th
        self.nlm = nlm
        self.a = np.zeros((nlm+1,nlm+1))
        for l in range(self.nlm+1):
            for m in range(l+1):
                if m == 0:
                    self.a[l,m] = 1
                else:
                    self.a[l,m] = (-1)**m*np.sqrt(2*math.factorial(l-m)/math.factorial(l+m))
    
    def baseline(self):
        """
        This method returns the "baseline" associated Legendre Polynomials, i.e.
        the surface across which they are calculated has its center in the coordinate 
        system the calculation is performed in.

        Parameters
        ----------
        None.

        Returns
        -------
        array
            Schmidt quasi-normalized associated Legendre Polynomials.
        array
            Differential of the Legendre Polyomials.

        """
        P = np.zeros((self.nlam,self.nth,self.nlm+1,self.nlm+1))
        dP = np.zeros((self.nlam,self.nth,self.nlm+1,self.nlm+1))
        for i in range(self.nth):
            for j in range(self.nlam):
                P[j,i], dP[j,i] = self.a.T*special.lpmn(self.nlm,self.nlm,np.cos(self.th[i]))
    
        return P.T, dP.T
    
    def transformedL(self, th):
        """
        This method is used when the Legendre Polynomials are to be calculated
        across a surface with a center that is offset from the origin of the
        coordinate system the Legendre Polynomials are calculated in. 

        Parameters
        ----------
        th : array
            Values for theta across the offset surface.

        Returns
        -------
        array
            Schmidt quasi-normalized associated Legendre Polynomials.
        array
            Differential of the Legendre Polyomials.

        """
        P = np.zeros((self.nlam,self.nth,self.nlm+1,self.nlm+1))
        dP = np.zeros((self.nlam,self.nth,self.nlm+1,self.nlm+1))
        for i in range(self.nth):
            for j in range(self.nlam):
                P[j,i], dP[j,i] = self.a.T*special.lpmn(self.nlm,self.nlm,np.cos(th[i,j]))
    
        return P.T, dP.T

    
    def Legendre_array(self, N, th):
        """        
        This method is used to calculate the Legendre polynomials in a 1D-array
        format, whereas the other methods return (nth)x(nlam) matrices.
        
        Parameters
        ----------
        N : int
            Number of grid points along the array. 
        th : array
            Values for theta long the array.

        Returns
        -------
        array
            Schmidt quasi-normalized associated Legendre Polynomials.
        array
            Differential of the Legendre Polyomials.

        """
        P = np.zeros((N,self.nlm+1,self.nlm+1))
        dP = np.zeros((N,self.nlm+1,self.nlm+1))
        for i in range(N):
            P[i], dP[i] = self.a.T*special.lpmn(self.nlm,self.nlm,np.cos(th[i]))
    
        return P.T, dP.T
    
    def Legendre_grid(self, N, th):
        """        
        This method is used to calculate the Legendre polynomials in a 1D-array
        format, whereas the other methods return (nth)x(nlam) matrices.
        
        Parameters
        ----------
        N : int
            Number of grid points along the array. 
        th : array
            Values for theta long the array.

        Returns
        -------
        array
            Schmidt quasi-normalized associated Legendre Polynomials.
        array
            Differential of the Legendre Polyomials.

        """
        P = np.zeros((N,N,N,self.nlm+1,self.nlm+1))
        dP = np.zeros((N,N,N,self.nlm+1,self.nlm+1))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    P[i,j,k], dP[i,j,k] = self.a.T*special.lpmn(self.nlm,self.nlm,np.cos(th[i,j,k]))
    
        return P.T, dP.T


