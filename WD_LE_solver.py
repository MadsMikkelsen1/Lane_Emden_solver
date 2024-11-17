import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.integrate as integrate
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
    dt = 0.001  # Smaller step size for better convergence
    max_steps = 50000  # Increased limit of steps to ensure longer solution
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
        t.append(t[-1] + dt)
        y1.append(max(0, y[0]))  # Ensure theta does not go below zero
        y2.append(y[1])
        i += 1
    return np.array(t), np.array(y1), np.array(y2)

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

    # Mass profile calculation using Simpson's rule for better accuracy
    mass = np.zeros_like(xi)
    for i in range(1, len(xi)):
        mass[i] = 4 * np.pi * rho_c * r_n**3 * integrate.simps(rho[:i] * xi[:i]**2, xi[:i])

    # Radius
    r = r_n * xi

    # Normalize r to solar radius range 0-1
    r_normalized = r / r_sun

    return r_normalized, mass / m_sun, rho

# Calculate and print radius and mass for the given central densities (in cgs units)
rho_c_values = [1e12, 5e12, 5e11]  # Central densities in g/cm^3
m_sun = 1.989e33  # Solar mass in g
r_sun = 6.957e10  # Solar radius in cm
n = 1.5  # Polytropic index for a white dwarf
K = 1.00e13  # Polytropic constant in cm^4 g^{-2/3} s^{-2}

# Calculate properties for each white dwarf
for i, rho_c in enumerate(rho_c_values):
    r, mass, rho = white_dwarf_properties(n, K, rho_c)
    print(f"\nWhite Dwarf {i + 1} with rho_c = {rho_c:.1e} g/cm^3:")
    print(f"Radius: {r[-1] * r_sun:.4e} cm")
    print(f"Mass: {mass[-1]:.4f} M_sun")

# Plot mass profiles for the white dwarfs
plt.figure(figsize=(10, 6))
for rho_c in rho_c_values:
    r, mass, _ = white_dwarf_properties(n, K, rho_c)
    plt.plot(r, mass, label=rf'$\rho_c = {rho_c:.1e} \; \mathrm{{g/cm^3}}$')
plt.xlabel(r'$r / R_\odot$')
plt.ylabel(r'$m / M_\odot$')
plt.legend()
plt.show()

# Plot density profiles for the white dwarfs
plt.figure(figsize=(10, 6))
for rho_c in rho_c_values:
    r, _, rho = white_dwarf_properties(n, K, rho_c)
    plt.plot(r, rho, label=rf'$\rho_c = {rho_c:.1e} \; \mathrm{{g/cm^3}}$')
plt.xlabel(r'$r / R_\odot$')
plt.ylabel(r'$\log(\rho) \; (\mathrm{g/cm^3})$')
plt.yscale('log')
plt.legend()
plt.show()

# Assume the Sun will end as a 0.5405 M_sun white dwarf
M_sun_wd = 0.5405 * m_sun  # Mass of the white dwarf in g

# Solve Lane-Emden equation for n = 1.5
xi, theta, d_theta = lane_emden_solver(n)
xi_1 = xi[-1]
d_theta_xi_1 = (theta[-1] - theta[-2]) / (xi[-1] - xi[-2]) if len(xi) > 2 else d_theta[-1]

# Correct calculation of central density using mass-radius relation derived from Lane-Emden
G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
r_n = (K * (n + 1) / (4 * np.pi * G))**(1 / (3 - n))

# Calculate central density using the provided formula
# M = -4 * pi * r_n^3 * rho_c * xi_1^2 * d_theta/d_xi evaluated at xi_1
rho_c_sun = -M_sun_wd / (4 * np.pi * r_n**3 * xi_1**2 * d_theta_xi_1)
radius_sun_wd = r_n * xi_1

print(f"\nWhite Dwarf with Mass = 0.5405 M_sun:")
print(f"Central Density: {rho_c_sun:.4e} g/cm^3")
print(f"Radius: {radius_sun_wd:.4e} cm")

# Plot log rho and m/M_sun as a function of r/R_sun for the Sun's white dwarf
r, mass, rho = white_dwarf_properties(n, K, rho_c_sun)

plt.figure(figsize=(10, 6))
plt.plot(r, mass, label=r'$m (M_\odot)$')
plt.xlabel(r'$r / R_\odot$')
plt.ylabel(r'$m (M_\odot)$')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(r, rho, label=r'$\log(\rho) \; (\mathrm{g/cm^3})$')
plt.xlabel(r'$r / R_\odot$')
plt.ylabel(r'$\log(\rho) \; (\mathrm{g/cm^3})$')
plt.yscale('log')
plt.show()