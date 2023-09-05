# Mutual Induction Coupling
This repository contains the numerical implementation of the analytical model to solve the coupled induction between two spherical conductors, e.g., an ocean and a reservoir.
This code is divided into different modules that provide essential functions and are required to load.

# Usage
The initial setup occurs in Parameters.py. This is where the geometrical information (i.e., reservoir position/size, ocean thickness/depth) is stored, as well as induction
parameters, such as conductivity,frequency of the background field, and amplitude of the background field. An exemple that solves the coupled induction is included in
General.py. The primary return here are the ocean's and reservoir's internal Gauss coefficients of interation step $n$, with which the internal magnetic fields can be calculated
using the methods provided by MagneticField.py.


