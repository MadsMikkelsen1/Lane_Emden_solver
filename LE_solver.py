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
    v[1] = (y[0]**n) * t**2 if y[0] >= 0 else 0  # Prevent invalid value in power
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

# Plot the results for n = 0 to 5 in steps of 0.5
plt.figure(figsize=(10.24, 7.68))
n_values = np.arange(0, 5.5, 0.5)

# Store xi values where theta hits zero and related parameters
xi_values_at_theta_zero = []
results = []

for n in n_values:
    xi, theta, d_theta = lane_emden_solver(n)
    plt.plot(xi, theta, label=f'n = {n}')
    if len(xi) > 0:
        xi_1 = xi[-1]
        # Calculate d_theta/d_xi at xi_1 using backward difference
        if len(xi) > 2:
            d_theta_xi_1 = (theta[-1] - theta[-2]) / (xi[-1] - xi[-2])
        else:
            d_theta_xi_1 = d_theta[-1]  # Fallback if there aren't enough points
        xi_values_at_theta_zero.append((n, xi_1))
        if n != 0 and n != 5:  # Skip special cases for n=0 and n=5 in calculations
            # Recalculating D_n, M_n, R_n, and B_n with updated formulas
            D_n = -xi_1 / (3 * d_theta_xi_1)
            M_n = -xi_1**2 * d_theta_xi_1
            R_n = xi_1  # Stellar radius corresponds to xi_1
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
print(f"{'Poly Index (n)':<20}{'Xi at Theta=0':<20}")
print("-" * 40)
for n, xi_zero in xi_values_at_theta_zero:
    print(f"{n:<20}{xi_zero:<20.4f}")

# Print a table of D_n, M_n, R_n, and B_n values
print(f"\n{'Poly Index (n)':<20}{'D_n':<20}{'M_n':<20}{'R_n':<20}{'B_n':<20}")
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