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

## Task 3 ##
## Compute D_n, R_n, M_n and B_n for each n. Careful with cases n = 0 and n = 5 // Choose constants carefully ##
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
plt.xlabel(r'$\log(r / R_{\odot})$')
plt.ylabel(r'$\log(\theta(\xi))$')
plt.ylim([-0.25, 1.05])
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

## Task 4 ##
## Calculate the radius and mass of non-relativistic white dwarf (polytropic index n = 3/2), which composition µ_e = 2, ##
## central densities: 1) ρ_c = 10^9 kg m^{-3}, 2) ρ_c = 5 x 10^9 kg m^{-3} and 3) ρ_c = 5 x 10^8 kg m^{-3} ##
def calculate_polytropic_constant(n, M, R, xi_1, d_theta_xi_1, mu_e=2):
    G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
    m_u = 1.66053906660e-24  # Atomic mass unit in g
    m_e = 9.10938356e-28  # Mass of electron in g

    term1 = G / (n + 1)
    term2 = np.power(M, 1 - 1 / n) * np.power(R, -1 + 3 / n)
    term3 = 4 * np.pi / (np.power(xi_1, n + 1) * np.power(-d_theta_xi_1, n - 1))
    K = term1 * term2 * np.power(term3, 1 / n)

    return K

# Function to calculate properties of a white dwarf for a given central density (in cgs units)
def white_dwarf_properties(n, K, rho_c, mu_e=2):
    G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
    m_u = 1.66053906660e-24  # Atomic mass unit in g
    m_e = 9.10938356e-28  # Mass of electron in g

    # Use Lane-Emden solver to solve for polytropic index n
    xi, theta, d_theta = lane_emden_solver(n)
    xi_1 = xi[-1]
    d_theta_xi_1 = (theta[-1] - theta[-2]) / (xi[-1] - xi[-2]) if len(xi) > 2 else d_theta[-1]

    # Central pressure and density profile
    P_c = K * np.power(rho_c, 1 + 1 / n)
    rho = rho_c * np.power(theta, n)

    # Length scale
    term1 = (n + 1) * P_c
    term2 = 4 * np.pi * G * np.power(rho_c, 2)
    r_n = np.sqrt(term1 / term2)

    # Mass profile calculation
    mass = np.zeros_like(xi)
    for i in range(1, len(xi)):
        mass[i] = 4 * np.pi * rho_c * r_n**3 * np.trapz(xi[:i]**2 * theta[:i], xi[:i])

    # Radius
    r = r_n * xi

    # Normalize r to solar radius range 0-1
    r_normalized = r / r[-1] if r[-1] != 0 else r

    return r_normalized, mass, rho


## Task 5 ##
## Plot log ρ and m/M_{\odot} as a function of r/R_{\odot} for white dwarves wih the three densities from Task 4 ##

# Calculate and print radius and mass for the given central densities (in cgs units)
rho_c_values = [1e12, 5e12, 5e11]  # Central densities in g/cm^3
m_sun = 1.989e33  # Solar mass in g
r_sun = 6.957e10  # Solar radius in cm
n = 1.5  # Polytropic index for a white dwarf

# Calculate K for each white dwarf
for i, rho_c in enumerate(rho_c_values):
    xi, theta, d_theta = lane_emden_solver(n)
    xi_1 = xi[-1]
    d_theta_xi_1 = (theta[-1] - theta[-2]) / (xi[-1] - xi[-2]) if len(xi) > 2 else d_theta[-1]
    K = calculate_polytropic_constant(n, M=m_sun, R=r_sun, xi_1=xi_1, d_theta_xi_1=d_theta_xi_1)
    r, mass, rho = white_dwarf_properties(n, K, rho_c)
    print(f"\nWhite Dwarf {i + 1} with rho_c = {rho_c:.1e} g/cm^3:")
    print(f"Radius: {r[-1] * r_sun:.4e} cm")
    print(f"Mass: {mass[-1] / m_sun:.4f} M_sun")

# Plot mass profiles for the white dwarfs
plt.figure(figsize=(10, 6))
for rho_c in rho_c_values:
    xi, theta, d_theta = lane_emden_solver(n)
    xi_1 = xi[-1]
    d_theta_xi_1 = (theta[-1] - theta[-2]) / (xi[-1] - xi[-2]) if len(xi) > 2 else d_theta[-1]
    K = calculate_polytropic_constant(n, M=m_sun, R=r_sun, xi_1=xi_1, d_theta_xi_1=d_theta_xi_1)
    r, mass, _ = white_dwarf_properties(n, K, rho_c)
    plt.plot(r, mass / m_sun, label=rf'$\rho_c = {rho_c:.1e} \; \mathrm{{g/cm^3}}$')
plt.xlabel(r'$r / R_\odot$')
plt.ylabel(r'$m / M_\odot$')
plt.legend()
plt.show()

# Plot density profiles for the white dwarfs
plt.figure(figsize=(10, 6))
for rho_c in rho_c_values:
    xi, theta, d_theta = lane_emden_solver(n)
    xi_1 = xi[-1]
    d_theta_xi_1 = (theta[-1] - theta[-2]) / (xi[-1] - xi[-2]) if len(xi) > 2 else d_theta[-1]
    K = calculate_polytropic_constant(n, M=m_sun, R=r_sun, xi_1=xi_1, d_theta_xi_1=d_theta_xi_1)
    r, _, rho = white_dwarf_properties(n, K, rho_c)
    plt.plot(r, rho, label=rf'$\rho_c = {rho_c:.1e} \; \mathrm{{g/cm^3}}$')
plt.xlabel(r'$r / R_\odot$')
plt.ylabel(r'$\log(\rho) \; (\mathrm{g/cm^3})$')
plt.yscale('log')
plt.legend()
plt.show()

## Task 6 ##
## Assume the Sun will end its life as a 0.5405 M_{\odot} white dwarf (Schröder and Smith 2008). Find the central density of ##
## that white dwarf, its expected radius, and plot log ρ and m/M_{\odot} as a function of r/R_{\odot}. ##

# Assume the Sun will end as a 0.5405 M_sun white dwarf
M_sun_wd = 0.5405 * m_sun  # Mass of the white dwarf in g

# Solve Lane-Emden equation for n = 1.5
xi, theta, d_theta = lane_emden_solver(n)
xi_1 = xi[-1]
d_theta_xi_1 = (theta[-1] - theta[-2]) / (xi[-1] - xi[-2]) if len(xi) > 2 else d_theta[-1]

# Calculate polytropic constant K for the Sun's white dwarf
K = calculate_polytropic_constant(n, M=M_sun_wd, R=r_sun, xi_1=xi_1, d_theta_xi_1=d_theta_xi_1)

# Calculate central density using the mass-radius relation derived from Lane-Emden
G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
r_n = (K * (n + 1) / (4 * np.pi * G))**(1 / (3 - n))

# Calculate radius and central density for the white dwarf
rho_c_sun = -M_sun_wd / (4 * np.pi * r_n**3 * xi_1**2 * d_theta_xi_1)
radius_sun_wd = r_n * xi_1

print(f"\nWhite Dwarf with Mass = 0.5405 M_sun:")
print(f"Central Density: {rho_c_sun:.4e} g/cm^3")
print(f"Radius: {radius_sun_wd:.4e} cm")

# Plot log rho and m/M_sun as a function of r/R_sun for the Sun's white dwarf
r, mass, rho = white_dwarf_properties(n, K, rho_c_sun)

plt.figure(figsize=(10, 6))
plt.plot(r, mass / m_sun, label=r'$m (M_\odot)$')
plt.xlabel(r'$r / R_\odot$')
plt.ylabel(r'$m (M_\odot)$')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(r, rho, label=r'$\log(\rho) \; (\mathrm{g/cm^3})$')
plt.xlabel(r'$r / R_\odot$')
plt.ylabel(r'$\log(\rho) \; (\mathrm{g/cm^3})$')
plt.yscale('log')
plt.show()
