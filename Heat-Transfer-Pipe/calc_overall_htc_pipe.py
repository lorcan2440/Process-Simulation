'''
Heat transfer from a hot pipe to a cold water bath. 
'''

from matplotlib import pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI

plt.style.use(r'C:\LibsAndApps\Python config files\proplot_style.mplstyle')

# Constants
GRAVITY = 9.81  # gravitational acceleration [m s^-2]

# Boundary temperatures
T_STEAM = 400  # temperature of steam inside pipe [K]
T_WATER = 300  # temperature of water outside pipe [K]

# Geometry
R_PIPE_INNER = 0.1  # internal radius of pipe [m]
L_PIPE = 1  # length of pipe [m]
R_PIPE_OUTER = 0.11  # external radius of pipe [m]

# Flow properties
Q = 10  # volumetric flow rate [m^3 s^-1]

# Heat transfer coefficient functions
# HTC between steam and pipe [W m^-2 K^-1]: Dittus-Boelter correlation for forced convection in a cooled pipe
h_in = lambda Re, Pr: 0.023 * Re ** (4/5) * Pr ** 0.3
# HTC between pipe and water [W m^-2 K^-1]: Churchill-Chu correlation for natural convection from a cylinder
h_out = lambda Ra, Pr: (0.6 + 0.387 * Ra ** (1/6) / (1 + (0.559 / Pr) ** (9/16)) ** (8/27)) ** 2

# Steam properties
rho_steam = lambda T: PropsSI('D', 'T|gas', T, 'P', 1e5, 'Water')  # steam density [kg m^-3]
mu_steam = lambda T: PropsSI('V', 'T|gas', T, 'P', 1e5, 'Water')  # steam viscosity [Pa s]
k_steam = lambda T: PropsSI('L', 'T|gas', T, 'P', 1e5, 'Water')  # steam thermal conductivity [W m^-1 K^-1]
cp_steam = lambda T: PropsSI('C', 'T|gas', T, 'P', 1e5, 'Water')  # steam specific heat capacity [J kg^-1 K^-1]
Pr_steam = lambda T: PropsSI('Prandtl', 'T|gas', T, 'P', 1e5, 'Water')  # steam Prandtl number [-]

# Pipe properties
k_pipe = 50  # thermal conductivity of pipe [W m^-1 K^-1]


def R_total(T1: float, T2: float, print_props: bool = False) -> list[float]:
    '''
    Calculate the thermal resistance of the pipe.
    
    ### Arguments
    #### Required
    - `T1` (float): temperature of the inner steam-pipe boundary [K]
    - `T2` (float): temperature of the outer pipe-water boundary [K]

    #### Optional
    - `print_props` (bool, default = False): whether to print the properties used in the calculation
    
    ### Returns
    - `list[float]`: thermal resistances of the inner, outer and total pipe sections [K W^-1]
    '''

    # Water properties
    rho_water = lambda T: PropsSI('D', 'T|liquid', T, 'P', 1e5, 'Water')  # water density [kg m^-3]
    mu_water = lambda T: PropsSI('V', 'T|liquid', T, 'P', 1e5, 'Water')  # water viscosity [Pa s]
    k_water = lambda T: PropsSI('L', 'T|liquid', T, 'P', 1e5, 'Water')  # water thermal conductivity [W m^-1 K^-1]
    cp_water = lambda T: PropsSI('C', 'T|liquid', T, 'P', 1e5, 'Water')  # water specific heat capacity [J kg^-1 K^-1]
    Pr_water = lambda T: PropsSI('Prandtl', 'T|liquid', T, 'P', 1e5, 'Water')  # water Prandtl number [-]
    alpha_water = lambda T: k_water(T) / (rho_water(T) * cp_water(T))  # water thermal diffusivity [m^2 s^-1]

    # Reynolds number of steam = rho V D / mu
    Re_steam = lambda T: (rho_steam(T) * 2 * Q) / (mu_steam(T) * np.pi * R_PIPE_INNER)
    # Rayleigh number of water = g beta (T_s - T_inf) D^3 / (nu^2 alpha)
    Ra_water = lambda T: (rho_water(T_WATER) - rho_water(T)) * (2 * R_PIPE_OUTER) ** 3 * GRAVITY / (mu_water(T) * alpha_water(T))

    # Thermal resistances
    R_in = lambda T: 1 / (h_in(Re_steam(T), Pr_steam(T)) * 2 * np.pi * R_PIPE_INNER * L_PIPE)  # T: film temperature
    R_pipe = np.log(R_PIPE_OUTER / R_PIPE_INNER) / (2 * np.pi * k_pipe * L_PIPE)
    R_out = lambda T: 1 / (h_out(Ra_water(T), Pr_water(T)) * 2 * np.pi * R_PIPE_OUTER * L_PIPE)  # T: film temperature

    # Film temperatures
    T_film_in = (T1 + T_STEAM) / 2
    T_film_out = (T2 + T_WATER) / 2

    if print_props:
        print(f'T_film_in = {T_film_in}, T_film_out = {T_film_out}')
        print('Re_steam = ', Re_steam(T_film_in))
        print('Pr_steam = ', Pr_steam(T_film_in))
        print('Ra_water = ', Ra_water(T_film_out))
        print('Pr_water = ', Pr_water(T_film_out))
        print('h_in = ', h_in(Re_steam(T_film_in), Pr_steam(T_film_in)))
        print('h_out = ', h_out(Ra_water(T_film_out), Pr_water(T_film_out)))
        print('R_in = ', R_in(T_film_in))
        print('R_pipe = ', R_pipe)
        print('R_out = ', R_out(T_film_out))
        print('R_total = ', R_in(T_film_in) + R_pipe + R_out(T_film_out))

    return R_in(T_film_in), R_out(T_film_out), R_in(T_film_in) + R_pipe + R_out(T_film_out)

# Calculate heat transfer rate using iterative solution, recalculating fluid properties at new temperatures
# Initial guesses for boundary temperatures: T1 = T_STEAM and T2 = T_WATER
T1 = 400  # could also just use T_STEAM
T2 = 350  # could also just use T_WATER
props = {'T1': [], 'T2': [], 'Q_dot': [], 'R_total': [], 'U_overall': []}
for i in range(10):
    R_in, R_out, R_tot = R_total(T1, T2)
    U_overall = 1 / (2 * np.pi * R_PIPE_OUTER * L_PIPE * R_tot)
    Q_dot = (T_STEAM - T_WATER) / R_tot
    props['T1'].append(T1)
    props['T2'].append(T2)
    props['Q_dot'].append(Q_dot)
    props['R_total'].append(R_tot)
    props['U_overall'].append(U_overall)
    print(f'T_1 = {T1}, T_2 = {T2}, Q_dot = {Q_dot}, R_total = {R_tot}, U_overall = {U_overall}')
    print('----------------')
    # Update T1 and T2 and recalculate properties
    T1 = T_STEAM - Q_dot * R_in
    T2 = T_WATER + Q_dot * R_out

# Plot convergence results in 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
axs[0, 0].plot(props['T1'], label=r'$T_1$ (inner wall)')
axs[0, 0].plot(props['T2'], label=r'$T_2$ (outer wall)')
axs[0, 0].set_title(r'$T_1$ and $T_2$: Wall temperatures')
axs[0, 0].set_xlabel('Iteration #')
axs[0, 0].set_ylabel('Wall temperature [K]')
axs[0, 0].legend()

axs[0, 1].plot(props['Q_dot'])
axs[0, 1].set_title(r'$\dot{Q}$: heat transfer rate')
axs[0, 1].set_xlabel('Iteration #')
axs[0, 1].set_ylabel('Heat transfer rate [W]')

axs[1, 0].plot(props['R_total'])
axs[1, 0].set_title(r'$R_{total}$: total thermal resistance')
axs[1, 0].set_xlabel('Iteration #')
axs[1, 0].set_ylabel(r'Thermal resistance [$K W^{-1}$]')
#axs[1, 0].set_yscale('log')

axs[1, 1].plot(props['U_overall'])
axs[1, 1].set_title(r'$U_{overall}$: overall heat transfer coefficient')
axs[1, 1].set_xlabel('Iteration #')
axs[1, 1].set_ylabel(r'Overall HTC (outer wall) [$W m^{-2} K^{-1}$]')
#axs[1, 1].set_yscale('log')

for ax in axs.flat:
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(plt.NullLocator())

fig.suptitle('Convergence of iterative solution', fontsize=14)

# vertical space
fig.tight_layout(pad=2)
plt.show()