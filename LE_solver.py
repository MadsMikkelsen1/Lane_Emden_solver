import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns

## Task 1 ## 
## Solve LE eqn. for n = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] and plot the results ##

## Task 2 ##
## Compare with the two cases where we have an analytical solution (n = 0, 1) ##

## Task 3 ##
## Compute D_n, R_n, M_n and B_n for each n. Careful with cases n = 0 and n = 5 // Choose constants carefully ##

## Task 4 ##
## Calculate the radius and mass of non-relativistic white dwarf (polytropic index n = 3/2), which composition µ_e = 2, ##
## central densities: 1) ρ_c = 10^9 kg m^{-3}, 2) ρ_c = 5 x 10^9 kg m^{-3} and 3) ρ_c = 5 x 10^8 kg m^{-3} ##

## Task 5 ##
## Plot log ρ and m/M_{\odot} as a function of r/R_{\odot} for white dwarves wih the three densities from Task 4 ##

## Task 6 ##
## Assume the Sun will end its life as a 0.5405 M_{\odot} white dwarf (Schröder and Smith 2008). Find the central density of ##
## that white dwarf, its expected radius, and plot log ρ and m/M_{\odot} as a function of r/R_{\odot}. ##