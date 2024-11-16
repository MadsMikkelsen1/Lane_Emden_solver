import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns
from IPython.display import display, Math
sns.set_theme(style="darkgrid")

# Define the Runge-Kutta 4th Order Integrator
def runkutt(y, t, dt, n):
    c1 = g(y, t, n)
    c2 = g(y + dt * c1 / 2, t + dt / 2, n)
    c3 = g(y + dt * c2 / 2, t + dt / 2, n)
    c4 = g(y + dt * c3, t + dt, n)
    return y + dt * (c1 + 2 * c2 + 2 * c3 + c4) / 6

# Function defining the differential equations
def g(y, t, n):
    v = np.zeros(2)
    v[0] = -y[1] / t**2 if t != 0 else 0  # Prevent division by zero
    v[1] = (y[0]**n) * t**2 if y[0] >= 0 else 0
    return v

# Solve the Lane-Emden equation for a given polytropic index n
def lane_emden_solver(n):
    dt = 0.01  # Smaller step size for better convergence
    max_steps = 10000  # Limit the number of steps to avoid infinite loops
    y1 = []
    y2 = []
    y = np.zeros(2)
    y[0] = 1  # Initial condition for theta
    y[1] = 0  # Initial condition for theta'

    y1.append(y[0])
    y2.append(y[1])
    t = [0]
    i = 1
    while y[0] > 0 and i < max_steps:
        y = runkutt(y, t[-1], dt, n)
        if y[0] < 0:
            break
        t.append(t[-1] + dt)
        y1.append(y[0])
        y2.append(y[1])
        i += 1
    return np.array(t), np.array(y1), np.array(y2)

plt.figure(figsize=(12, 8))
n_values = np.arange(0, 5.5, 0.5)

# Store xi values where theta hits zero and related parameters
xi_values_at_theta_zero = []
results = []

# Constants
m_sun = 1.989e30  # Solar mass in kg
r_sun = 6.957e8   # Solar radius in meters
G = 6.67430e-11   # Gravitational constant

for n in n_values:
    xi, theta, d_theta = lane_emden_solver(n)
    plt.plot(xi, theta, label=f'n = {n}')
    if len(xi) > 0:
        xi_1 = xi[-1]
        d_theta_xi_1 = d_theta[-1]
        xi_values_at_theta_zero.append((n, xi_1))
        if n != 0 and n != 5:  # Skip special cases for n=0 and n=5 in calculations
            # Calculating K
            term1 = G / (n + 1)
            term2 = np.power(1.0 * m_sun, 1 - 1 / n) * np.power(1.0 * r_sun, -1 + 3 / n)  # Using M = 1 M_sun, R = 1 R_sun
            term3 = 4 * np.pi
            term4 = np.power(xi_1, n + 1) * np.power(d_theta_xi_1, n - 1)
            K = term1 * term2 * np.power(term3 / term4, 1 / n)

            # Calculating the central pressure and pressure profile
            term1_p = 8.952e+14 * np.power(1.0, 2) * np.power(1.0, -4)  # Using M = 1 M_sun, R = 1 R_sun
            term2_p = (n + 1) * np.power(d_theta_xi_1, 2)
            P_c = term1_p / term2_p  # dyne/cm^2
            P = P_c * np.power(theta, n + 1)

            # Calculating central density and density profile
            rho_c = np.power(P_c / K, n / (n + 1))
            rho = rho_c * np.power(theta, n)

            D_n = (-3 / (xi_1 * d_theta_xi_1))**-1
            M_n = -xi_1**2 * d_theta_xi_1
            alpha = ((n + 1) * K / (4 * np.pi * G * rho_c**((n - 1) / n)))**0.5
            R_n = alpha * xi_1
            B_n = (1 / (n + 1)) * (-xi_1**2 * d_theta_xi_1)**(-2 / 3)
            results.append((n, D_n, M_n, R_n, B_n))

plt.hlines(0, 0, 100, color="black")
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\Theta(\xi)$')
plt.ylim([-0.5, 1.05])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 18
plt.legend()
plt.show()

# Print a table of xi values where theta hits zero
print(f"{'Polytropic Index (n)':<20}{'Xi at Theta=0':<20}")
print("-" * 40)
for n, xi_zero in xi_values_at_theta_zero:
    print(f"{n:<20}{xi_zero:<20.4f}")

# Print a table of D_n, M_n, R_n, and B_n values
print(f"\n{'Polytropic Index (n)':<20}{'D_n':<20}{'M_n':<20}{'R_n':<20}{'B_n':<20}")
print("-" * 100)
for n, D_n, M_n, R_n, B_n in results:
    print(f"{n:<20}{D_n:<20.4f}{M_n:<20.4f}{R_n:<20.4f}{B_n:<20.4f}")


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