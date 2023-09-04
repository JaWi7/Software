# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:15:30 2022

"""

import numpy as np

class MagneticField(object):
    def __init__(self, nth, nlam, nlm, omega):
        """
        Defines the MagneticField class, used for any calculations that pertain
        to the induced and inducing magnetic fields. Magnetic fields and 
        Gauss coefficients are required to be in nT.

        Parameters
        ----------
        nth : int
            Number of grid points along latitude.
        nlam : int
            Number of grid points along longitude.
        nlm : int
            Maximum number of degrees/orders.
        omega : TYPE
            Frequency of the inducing field.

        Returns
        -------
        None.

        """
        self.nth = nth
        self.nlam = nlam
        self.nlm = nlm+1
        self.omega = omega
        
    def InternalField(self, grid, nlm, g, h, P):
        """
        Calculate the radial component of the internal (i.e., induced) field of 
        ocean or reservoir across it's own surface.

        Parameters
        ----------
        grid : class
            Defines the grid across the surface using Grid.py.
        nlm : int
            Maximum number of degrees/orders.
        g : array
            Internal Gauss coefficients g_l^m.
        h : array
            Internal Gauss coefficients h_l^m.
        P : array
            Schmidt quasi-normalized Associated Legendre Polynomials, calculated
            using LegendrePolynomials.py.

        Returns
        -------
        array
            Radial component of the internal field across the surface grid.

        """
        self.b_int = np.zeros((self.nth, self.nlam))
        for l in range(1, nlm+1):
            for m in range(0, l+1):
                self.b_int += (l+1) * P[l,m] * (g[l,m] * np.cos(m*grid.LAM) + h[l,m] * np.sin(m*grid.LAM))
                                                        
        return self.b_int
    
    def ExternalField(self, grid, nlm, q, s, P):
        """
        Calculate radial component of the external (i.e., inducing) field on the
        surface of the body the external field is acting on. This method
        can be used to compare the inducing field resulting from external Gauss
        coefficients to the field resulting from the InternalTransformed method.

        Parameters
        ----------
        grid : class
            Defines the grid across the surface using Grid.py.
        nlm : int
            Maximum number of degrees/orders.
        q : array
            External Gauss coefficients q_l^m.
        s : array
            External Gauss coefficients s_l^m.
        P : array
            Schmidt quasi-normalized Associated Legendre Polynomials, calculated
            using LegendrePolynomials.py.

        Returns
        -------
        array
            Radial component of the external field across the surface grid.

        """

        self.b_ext = np.zeros((self.nth, self.nlam))
        for l in range(1, nlm+1):
            for m in range(0, l+1):
                self.bfield -= l * P[l,m] * (q[l,m] * np.cos(m*grid.LAM) + s[l,m] * np.sin(m*grid.LAM))
        return self.b_ext
    
    def InternalTransformed(self, nlm, grid, g, h, r, d, th, lam, P, dP):
        """
        Calculates the radial component of the internal field of ocean/reservoir
        across the surface of reservoir in reservoir-centered coordinates
        or vice versa. To do so, spherical components of the internal field are
        calculated in ocean-centered coordinates and transformed in reservoir-centered
        cartesian coordinates. From there, the radial component is calcualted,
        which is used to obtain the external Gauss coefficients of this inducing
        field (equation 9 in Winkenstern et al., 2023).
        
        Parameters
        ----------
        nlm : int
            Maximum number of degrees/orders.
        grid : class
            Defines the grid across the surface using Grid.py.
        g : array
            Internal Gauss coefficients g_l^m.
        h : array
            Internal Gauss coefficients h_l^m.
        r : float
            Outer radius of the ocean (or reservoir radius), in m.
        d : array
            Radial component of the reservoir surface grid points in ocean-centered
            coordinartes (or vice versa), in m.
        th : array
            Latitudinal component of the reservoir surface grid points in ocean-centered
            coordinartes (or vice versa), in rad.
        lam : array
            Longitudinal component of the reservoir surface grid points in ocean-centered
            coordinartes (or vice versa), in rad.
        P : array
            Schmidt quasi-normalized Associated Legendre Polynomials, calculated
            using LegendrePolynomials.py.
        dP : array
            Differential of the Schmidt quasi-normalized Associated Legendre 
            Polynomials, calculated using LegendrePolynomials.py.

        Returns
        -------
        array
            Transformed radial component of the internal field.

        """
        self.nlm = nlm+1
        self.b_r = np.zeros((self.nth, self.nlam))
        self.b_th = np.zeros((self.nth, self.nlam))
        self.b_lam = np.zeros((self.nth, self.nlam))
        
        for l in range(1, self.nlm):
            for m in range(0, l+1):
                self.b_r += (l+1)*(r/d)**(l+2) * P[l][m] * (
                    g[l,m] * np.cos(m*lam) + h[l,m] * np.sin(m*lam))
                self.b_th -= (r/d)**(l+2) * dP[l][m] * (
                    g[l,m] * np.cos(m*lam) + h[l,m] * np.sin(m*lam))
                self.b_lam += 1/(np.sin(th)+1e-28) * (r/d)**(l+2) * m * P[l][m] * (
                    g[l,m] * np.sin(m*lam) - h[l,m] * np.cos(m*lam))
        self.bx = (self.b_r * np.sin(th) * np.cos(lam) + self.b_th * np.cos(th) * np.cos(lam) - 
                  self.b_lam * np.sin(lam))
        self.by = (self.b_r * np.sin(th) * np.sin(lam) + self.b_th * np.cos(th) * np.sin(lam) +
                  self.b_lam * np.cos(lam))
        self.bz = self.b_r * np.cos(th) - self.b_th * np.sin(th)
        self.b_r_tr = (self.bx * np.sin(grid.TH) * np.cos(grid.LAM) + self.by * np.sin(grid.TH) * np.sin(grid.LAM) + 
                        self.bz * np.cos(grid.TH))
        return self.b_r_tr
    
    def calculate_int_l(self, l, grid, g, h, r, d, th, lam, P, dP):
        """
        Calculates the radial component of the internal multipole contribution 
        of degree l of ocean/reservoir across the surface of reservoir in 
        reservoir-centered coordinates or vice versa. Separate consideration
        becomes necessary when considering a non-zero phase shift, as the 
        time at which the field is induced changes for each multipole moment.
        This radial component is used to calculate the external Gauss coefficients
        induced by multipole degree l, the "final" external coefficients are the
        sum over all degrees up to lmax (nlm).

        Parameters
        ----------
        l : int
            Multipole moment l that is considered in this calculation.
        grid : class
            Defines the grid across the surface using Grid.py.
        g : array
            Internal Gauss coefficients g_l^m.
        h : array
            Internal Gauss coefficients h_l^m.
        r : float
            Outer radius of the ocean (or reservoir radius), in m.
        d : array
            Radial component of the reservoir surface grid points in ocean-centered
            coordinartes (or vice versa), in m.
        th : array
            Latitudinal component of the reservoir surface grid points in ocean-centered
            coordinartes (or vice versa), in rad.
        lam : array
            Longitudinal component of the reservoir surface grid points in ocean-centered
            coordinartes (or vice versa), in rad.
        P : array
            Schmidt quasi-normalized Associated Legendre Polynomials, calculated
            using LegendrePolynomials.py.
        dP : array
            Differential of the Schmidt quasi-normalized Associated Legendre 
            Polynomials, calculated using LegendrePolynomials.py.

        Returns
        -------
        TYPE
            Transformed radial component of the multipole degree l.

        """
        
        self.l = l
        self.b_r = np.zeros((self.nth, self.nlam))
        self.b_th = np.zeros((self.nth, self.nlam))
        self.b_lam = np.zeros((self.nth, self.nlam))
        
        for m in range(0, l+1):
            self.b_r += (l+1)*(r/d)**(l+2) * P[l][m] * (
                    g[l,m] * np.cos(m*lam) + h[l,m] * np.sin(m*lam))
            self.b_th -= (r/d)**(l+2) * dP[l][m] * (
                    g[l,m] * np.cos(m*lam) + h[l,m] * np.sin(m*lam))
            self.b_lam += 1/(np.sin(th)+1e-28) * (r/d)**(l+2) * m * P[l][m] * (
                    g[l,m] * np.sin(m*lam) - h[l,m] * np.cos(m*lam))
        self.bx = (self.b_r * np.sin(th) * np.cos(lam) + self.b_th * np.cos(th) * np.cos(lam) - 
                  self.b_lam * np.sin(lam))
        self.by = (self.b_r * np.sin(th) * np.sin(lam) + self.b_th * np.cos(th) * np.sin(lam) +
                  self.b_lam * np.cos(lam))
        self.bz = self.b_r * np.cos(th) - self.b_th * np.sin(th)
        self.b_r_tr = (self.bx * np.sin(grid.TH) * np.cos(grid.LAM) + self.by * np.sin(grid.TH) * np.sin(grid.LAM) + 
                        self.bz * np.cos(grid.TH))
        return self.b_r_tr
       
    def int_streamline(self, N, nlm, g, h, r, d, th, LAM, P, dP):
        """
        Calculates the Bx and By-components in the xy-plane, which can be used
        to e.g., visualize these components as a vector field.

        Parameters
        ----------
        N : int
            Number of grid points in x and y direction.
        nlm : int
            Maximum number of degrees/orders.
        g : array
            Internal Gauss coefficients g_l^m.
        h : array
            Internal Gauss coefficients h_l^m.
        r : float
            Outer radius of ocean (or reservoir radius), in m.
        d : array
            Radial component of the xy-grid in m.
        th : float
            Latitude of the xy-grid in rad.
        LAM : array
            Longitude of the xy-grid in rad.
        P : array
            Schmidt quasi-normalized Associated Legendre Polynomials, calculated
            using LegendrePolynomials.py.
        dP : array
            Differential of the Schmidt quasi-normalized Associated Legendre 
            Polynomials, calculated using LegendrePolynomials.py.

        Returns
        -------
        array
            Bx-component across the xy-grid.
        array
            By-component across the xy-grid.

        """
        
        self.nlm = nlm+1
        self.b_r = np.zeros((N, N))
        self.b_th = np.zeros((N, N))
        self.b_lam = np.zeros((N, N))
        
        for l in range(1, self.nlm):
            for m in range(0, l+1):
                self.b_r += np.real((l+1) * (r/d)**(l+2) * P[l][m] * (
                    g[l,m] * np.cos(m * LAM) + h[l,m] * np.sin(m * LAM)))
                self.b_th -= np.real((r/d)**(l+2) * dP[l][m] * (
                    g[l,m] * np.cos(m*LAM) + h[l,m] * np.sin(m*LAM)))
                self.b_lam += np.real(1/(np.sin(th)+1e-28) * (r/d)**(l+2) * m * P[l][m] * (
                    g[l,m] * np.sin(m*LAM) - h[l,m] * np.cos(m*LAM)) )
        self.bx = (self.b_r * np.sin(th) * np.cos(LAM) + self.b_th * np.cos(th) * np.cos(LAM) - 
                  self.b_lam * np.sin(LAM))
        self.by = (self.b_r * np.sin(th) * np.sin(LAM) + self.b_th * np.cos(th) * np.sin(LAM) +
                  self.b_lam * np.cos(LAM))
        
        return self.bx, self.by
    
    
    def int_flyby(self, N, nlm, g, h, r, d, th, lam, P, dP):
        """
        Calculates the internal magentic field in spherical coordinates along
        a given flyby trajectory. Also contains the magentic field in corresponding
        Cartesian coordinates.

        Parameters
        ----------
        N : int
            Number of points along trajectory.
        nlm : int
            Maximum number of degrees/orders.
        g : array
            Internal Gauss coefficients g_l^m.
        h : array
            Internal Gauss coefficients h_l^m.
        r : float
            Outer radius of ocean (or reservoir radius), in m.
        d : array
            Radial component of the xy-grid in m.
        th : float
            Latitude of the xy-grid in rad.
        LAM : array
            Longitude of the xy-grid in rad.
        P : array
            Schmidt quasi-normalized Associated Legendre Polynomials, calculated
            using LegendrePolynomials.py.
        dP : array
            Differential of the Schmidt quasi-normalized Associated Legendre 
            Polynomials, calculated using LegendrePolynomials.py.

        Returns
        -------
        array
            Br-component along the trajectory.
        array
            Bphi-component along the trajectory.
        array
            Btheta-component along the trajectory.
        array
            Bx-component along the trajectory.
        array
            By-component along the trajectory.
        array
            Bz-component along the trajectory.    

        """
        
        self.nlm=nlm+1
        self.b_r = np.zeros(N)
        self.b_lam = np.zeros(N)
        self.b_th = np.zeros(N)
        for l in range(1, self.nlm):
            for m in range(0, l+1):
                self.b_r += (l+1) * (r/d)**(l+2) * P[l][m] * (
                    g[l,m] * np.cos(m * lam) + h[l,m] * np.sin(m * lam))
                self.b_th -= (r/d)**(l+2) * dP[l][m] * (
                    g[l,m] * np.cos(m * lam) + h[l,m] * np.sin(m * lam))
                self.b_lam += 1/(np.sin(th)) * (r/d)**(l+2) * P[l][m] * (
                    g[l,m] * np.sin(m * lam) - h[l,m] * np.cos(m * lam))
        self.bx = (self.b_r * np.sin(th) * np.cos(lam) + self.b_th * np.cos(th) * np.cos(lam) - 
                  self.b_lam * np.sin(lam))
        self.by = (self.b_r * np.sin(th) * np.sin(lam) + self.b_th * np.cos(th) * np.sin(lam) +
                  self.b_lam * np.cos(lam))
        self.bz = self.b_r * np.cos(th) - self.b_th * np.sin(th)
                
        return self.b_r, self.b_lam, self.b_th, self.bx, self.by, self.bz
    
    def B_ind(self, B0x, B0y, phix, phiy, phi, t):
        """
        Calculates the inducing background fied at time t (plus additional phase
        shifts). This assumes the inducing field to be fully in the xy-plane.

        Parameters
        ----------
        B0x : float
            Amplitude of the Bx-component.
        B0y : TYPE
            Amplitude of the By-component.
        phix : TYPE
            Phase shift of the Bx-component.
        phiy : TYPE
            Phase shift of the By-component.
        phi : TYPE
            Phase shift of the induction response.
        t : TYPE
            DESCRIPTION.

        Returns
        -------
        B_indx : float
            Bx-component of the inducing field.
        B_indy : TYPE
            By-component of the inducing field.

        """
        
        B_indx = np.real(B0x * np.exp(1j*(self.omega*t+phix+phi)))
        B_indy = np.real(B0y * np.exp(1j*(self.omega*t+phiy+phi)))
        
        return B_indx, B_indy
                    