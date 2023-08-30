# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:14:15 2022

This class defines the Schmidt quasi-normalized assoociated Legendre 
polynomials. These are calculated using a recursion formula.

"""


import numpy as np 
import warnings
import matplotlib.pyplot as plt
from Grid import Grid

grid = Grid(180,360,(0, np.pi),(0,2*np.pi))


class LegendrePolynomials(object):
    def __init__(self, nlm):
        self.nlm = nlm+1
        self.alpha = np.zeros((self.nlm, self.nlm))
        self.beta = np.zeros((self.nlm, self.nlm))
        self.gamma = np.zeros((self.nlm, self.nlm))
        self.delta = np.zeros((self.nlm, self.nlm))
        for l in np.arange(0, self.nlm):
            for m in np.arange(0, l+1):
                if  m < l:
                    self.alpha[l,m] = 1/np.sqrt((l-m+1)*(l+m+1))
                    self.beta[l,m] = np.sqrt((l-m)*(l+m))/np.sqrt(((l-m+1)*(l+m+1)))
                    self.gamma[l,m] = np.sqrt(1/((l-m)*(l+m+1)))
                    self.delta[l,m] = np.sqrt((l+m))/np.sqrt(l+m+1)
                elif m == l:
                    self.alpha[l,m] = 1/np.sqrt((l-m+1)*(l+m+1))
                    self.beta[l,m] = np.sqrt((l-m)*(l+m))/np.sqrt((l-m+1)*(l+m+1))
                    self.delta[l,m] = np.sqrt((l+m))/np.sqrt(l+m+1)
        
    def baseline(self, grid):
        
        
        
        A = np.array([(2*l + 1) for l in range(0, self.nlm)])
        P = np.zeros((self.nlm, self.nlm, grid.nth))
        C = np.zeros((self.nlm, self.nlm, grid.nth, grid.nlam))
        S = np.zeros((self.nlm, self.nlm, grid.nth, grid.nlam))
        
        for l in range(0, self.nlm):
            if l == 0:
                for m in range(0, l+1):
                    if m == 0:
                        for i in range(0, grid.nth):
                            P[l,m,i] = 1
                            for j in range(0, grid.nlam):
                                C[l,m,i,j] = P[l,m,i]
            elif l == 1:
                for m in range(0,l+1):
                    if m == 0:
                        for i in range(0, grid.nth):
                            P[l,m,i] = np.cos(grid.th[i])
                            for j in range(0, grid.nlam):
                                C[l,m,i,j] = P[l,m,i]
                    elif m == 1:
                        for i in range(0, grid.nth):
                            P[l,m,i] = np.sqrt(1-np.cos(grid.th[i])**2)
                            for j in range(0, grid.nlam):
                                C[l,m,i,j] = P[l,m,i] * np.cos(m*grid.lam[j])
                                S[l,m,i,j] = P[l,m,i] * np.sin(m*grid.lam[j])
            else:
                for m in range(0, l+1):
                    if m == 0:
                        for i in range(0, grid.nth):
                            P[l,m,i] = 1/l * (A[l-1] * np.cos(grid.th[i]) * P[l-1,m,i] - (l-1) * P[l-2,m,i])
                            for j in range(0, grid.nlam):
                                C[l,m,i,j] = P[l,m,i]
                    elif m < l:
                        for i in range(0, grid.nth):
                            P[l,m,i] = (A[l-1] * self.alpha[l-1,m] * np.cos(grid.th[i]) * P[l-1,m,i] - 
                                            self.beta[l-1,m] * P[l-2,m,i])
                            for j in range(0, grid.nlam):
                                C[l,m,i,j] = P[l,m,i] * np.cos(m*grid.lam[j])
                                S[l,m,i,j] = P[l,m,i] * np.sin(m*grid.lam[j])
                    elif m == l:
                        for i in range(0, grid.nth):
                            P[l,m,i] = -1/(np.sqrt(1-np.cos(grid.th[i])**2)+1e-28) * (
                                    np.cos(grid.th[i]) * self.gamma[l,m-1] * P[l,m-1,i] - self.delta[l,m-1] * P[l-1,m-1,i])
                            for j in range(0, grid.nlam):
                                C[l,m,i,j] = P[l,m,i] * np.cos(m*grid.lam[j])
                                S[l,m,i,j] = P[l,m,i] * np.sin(m*grid.lam[j])

            print('Order {} done.'.format(l))
        return C, S
        
    def transformed_Legendre(self, grid, nlm_i, th, lam):
        
        self.th = th
        self.lam = lam
        
        A = np.array([(2*l + 1) for l in range(0, self.nlm)])
        self.P_tr = np.zeros((nlm_i+1, nlm_i+1, grid.nth, grid.nlam))
        self.P_tr_dif = np.zeros((nlm_i+1, nlm_i+1, grid.nth, grid.nlam))

        for l in range(0, nlm_i+1):
            if l == 0:
                for m in range(0, l+1):
                    if m == 0:
                        for i in range(0, grid.nth):
                            for j in range(0, grid.nlam):
                                self.P_tr[l,m,i,j] = 1
            elif l == 1:
                for m in range(0,l+1):
                    if m == 0:
                        for i in range(0, grid.nth):
                            for j in range(0, grid.nlam):
                                self.P_tr[l,m,i,j] = np.cos(self.th[i,j])
                                self.P_tr_dif[l,m,i,j] = -np.sin(self.th[i,j])
                    elif m == 1:
                        for i in range(0, grid.nth):
                            for j in range(0, grid.nlam):
                                self.P_tr[l,m,i,j] = np.sqrt(1-np.cos(self.th[i,j])**2)
                                self.P_tr_dif[l,m,i,j] = np.sin(self.th[i,j]) * np.cos(self.th[i,j]) /(
                                                        np.sqrt(1-np.cos(self.th[i,j])**2))
            else:
                for m in range(0, l+1):
                    if m == 0:
                        for i in range(0, grid.nth): 
                            for j in range(0, grid.nlam):
                                self.P_tr[l,m,i,j] = 1/l * (A[l-1] * np.cos(self.th[i,j]) * self.P_tr[l-1,m,i,j] - 
                                                            (l-1) * self.P_tr[l-2,m,i,j])
                                self.P_tr_dif[l,m,i,j] = 1/l * (A[l-1] * (np.cos(self.th[i,j]) * self.P_tr_dif[l-1,m,i,j] - 
                                                np.sin(self.th[i,j]) * self.P_tr[l-1,m,i,j]) - (l-1) * self.P_tr_dif[l-2,m,i,j])
                    elif m < l:
                        for i in range(0, grid.nth):
                            for j in range(0, grid.nlam):
                                self.P_tr[l,m,i,j] = (A[l-1] * self.alpha[l-1,m] * np.cos(self.th[i,j]) * self.P_tr[l-1,m,i,j] - 
                                            self.beta[l-1,m] * self.P_tr[l-2,m,i,j])
                                self.P_tr_dif[l,m,i,j] = (A[l-1] * self.alpha[l-1,m] * (np.cos(self.th[i,j]) * self.P_tr_dif[l-1,m,i,j] - 
                                            np.sin(self.th[i,j]) * self.P_tr[l-1,m,i,j]) - self.beta[l-1,m] * self.P_tr_dif[l-2,m,i,j])
                    elif m == l:
                        for i in range(0, grid.nth):
                            for j in range(0, grid.nlam):
                                self.P_tr[l,m,i,j] = -1/(np.sqrt(1-np.cos(self.th[i,j])**2)+1e-28) * (
                                    np.cos(self.th[i,j]) * self.gamma[l,m-1] * self.P_tr[l,m-1,i,j] - 
                                    self.delta[l,m-1] * self.P_tr[l-1,m-1,i,j])
                                self.P_tr_dif[l,m,i,j] = (-np.sin(self.th[i,j]) * np.cos(self.th[i,j])/
                                        ((1-np.cos(self.th[i,j])**2) + 1e-28) * self.P_tr[l,m,i,j] - 
                                        1/((np.sqrt(1-np.cos(self.th[i,j])**2)) + 1e-28) * (self.gamma[l,m-1] * (np.cos(self.th[i,j]) * 
                                        self.P_tr_dif[l,m-1,i,j] - np.sin(self.th[i,j]) * self.P_tr[l,m-1,i,j]) - 
                                        self.delta[l,m-1] * self.P_tr_dif[l-1,m-1,i,j]))

            print('Order {} done.'.format(l))
        return self.P_tr, self.P_tr_dif
    
    def Legendre_array(self, N, th):
        self.P_arr = np.zeros((self.nlm, self.nlm, N))
        self.P_arr_dif = np.zeros((self.nlm, self.nlm, N))
        
        A = np.array([(2*l + 1) for l in range(0, self.nlm)])
        
        for l in range(0, self.nlm):
            if l == 0:
                for m in range(0, l+1):
                    if m == 0:
                        for n in range(0, N):
                            self.P_arr[l,m,n] = 1
            elif l == 1:
                for m in range(0,l+1):
                    if m == 0:
                        self.P_arr[l,m] = np.cos(th)
                        self.P_arr_dif[l,m] = -np.sin(th)
                    elif m == 1:
                        self.P_arr[l,m] = np.sqrt(1-np.cos(th)**2)
                        self.P_arr_dif[l,m] = np.sin(th) * np.cos(th) / (np.sqrt(1-np.cos(th)**2))
            else:
                for m in range(0, l+1):
                    if m == 0:
                        self.P_arr[l,m] = 1/l * (A[l-1] * np.cos(th) * self.P_arr[l-1,m] - (l-1) * self.P_arr[l-2,m])
                        self.P_arr_dif[l,m] = 1/l * (A[l-1] * (np.cos(th) * self.P_arr_dif[l-1,m] - 
                                                np.sin(th) * self.P_arr[l-1,m]) - (l-1) * self.P_arr_dif[l-2,m])
                    elif m < l:
                        self.P_arr[l,m] = (A[l-1] * self.alpha[l-1,m] * np.cos(th) * self.P_arr[l-1,m] - 
                                            self.beta[l-1,m] * self.P_arr[l-2,m])
                        self.P_arr_dif[l,m] = (A[l-1] * self.alpha[l-1,m] * (np.cos(th) * self.P_arr_dif[l-1,m] - 
                                            np.sin(th) * self.P_arr[l-1,m]) - self.beta[l-1,m] * self.P_arr_dif[l-2,m])
                    elif m == l:
                        self.P_arr[l,m] = -1/(np.sqrt(1-np.cos(th)**2)+1e-28) * (
                            np.cos(th) * self.gamma[l,m-1] * self.P_arr[l,m-1] - 
                            self.delta[l,m-1] * self.P_arr[l-1,m-1])
                        self.P_arr_dif[l,m] = (-np.sin(th) * np.cos(th)/
                            ((1-np.cos(th)**2) + 1e-28) * self.P_arr[l,m] - 
                            1/((np.sqrt(1-np.cos(th)**2)) + 1e-28) * (self.gamma[l,m-1] * (np.cos(th) * 
                            self.P_arr_dif[l,m-1] - np.sin(th) * self.P_arr[l,m-1]) - 
                            self.delta[l,m-1] * self.P_arr_dif[l-1,m-1]))
        
        return self.P_arr, self.P_arr_dif


