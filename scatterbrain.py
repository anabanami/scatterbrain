from scipy.integrate import odeint
from scipy.special import struve, yn, riccati_jn, riccati_yn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150

# GLOBALS
# physical constants and SI units
hbar = 1.0545718e-34 # [Js]
epsilon = 8.854e-12 
au = 1.64877727436e-41 # [C^2 m^2 J^(-1)] 
e_charge = 1.60217662e-19 # [C]
eV = 1.602176634e-19 # [J]
meV = 1e-3 * eV
angstrom = 1e-10 # [m]

# table 1.
r_0 = 44.6814 * angstrom # [m]
mass_e = 9.10938356e-31 # [kg]
mass_eff_e = 0.47 * mass_e 
mass_eff_h = 0.54  * mass_e
mass_X = mass_eff_e + mass_eff_h
M_red = mass_eff_e * mass_X / (mass_eff_e + mass_X) # negative trion
# M_red = mass_eff_h * mass_X / (mass_eff_h + mass_X) # positive trion 

# table 2
r_star = 34 * angstrom  # only symmetric phase shift
v_0 = -58.5 * meV
alpha = 52 * (10**3) * au 

######################################################################
# Adjustable parameters
N =  2048
r = np.linspace(1e-6, 400, N) * angstrom
######################################################################

# POTENTIALS
def keldysh(r):
    # equation (2) 
    # [V]
    V_K = (np.pi / (2 * ( 4 * np.pi * epsilon) * r_0) * (struve(0, r / r_0) - yn(0, r / r_0)))
    return V_K

def V_Xe(r): 
    # equation (8)
    # [J]    
    r = np.array(r)
    dr = r_0 / 1000
    V_Xe = np.zeros_like(r)
    V_Xe[abs(r) <= r_star] = v_0 
    V_Xe[r > r_star] = - (alpha * e_charge **2 / 2) * ((keldysh(r[r > r_star] + dr) - keldysh(r[r > r_star])) / dr)**2
    return V_Xe


######################################################################
## Bound states search:
## SHOOTING method
## ENERGY
# E = -30 * meV #[J] initial guess

## ODE
# def ODE_shooting(r, y, E):
#     u, w = y
#     result = np.zeros_like(y)
#     if abs(u) > 1e100: # if blowing up return zero derivative
#         return result

#     else:
#         result[0] = w
#         result[1] = (- 1 / (4 * r**2) +  2 * M_red * (V_Xe(r) - E) / hbar**2) * u
#         return result

## ICS
# y_0 = [np.sqrt(r[0]), 1 / (2 * np.sqrt(r[0]))]

# E_lower = -31.15 * meV
# E_upper = -31.225 * meV

# signs = []
# E_Rmax = []
# new_bound = None

# # ODE solver
# for E in np.linspace(E_lower, E_upper, 2):
#     solution_ODE_shooting = solve_ivp(
                                        # lambda r, y: ODE_shooting(r, y, E),
                                        # [r[0], r[-1]], y_0, t_eval=r
                            # )
#     u = solution_ODE_shooting.y[0]
#     signs.append(np.sign(u[-1])) # storing the energy SIGN of solutions at Rmax
#     E_Rmax.append(u[-1]) # storing the energy VALUE of solutions at Rmax

# new_bound = (E_lower - (E_Rmax[0] - E_Rmax[1]) / 2 * meV)

# print(f"this is signs {signs}")
# print(f"this is E_Rmax {E_Rmax}")
# print(f"this is new_bound {new_bound}") 

# print(f"this is E_upper - E_lower = {E_upper - E_lower}")
# while E_upper - E_lower < 1e-6 * meV
 # or maybe 1e-6 * abs(E_min)


##     plt.plot(r / angstrom,  (u / abs(u[r < r_star]).max() * 5) + E / meV)
## plt.axis(ymin=v_0 / meV - 5, ymax=5)
## plt.ylabel(r'$E$ (meV) ')
## plt.xlabel(r'$r$ (Å)')
## plt.title('Shooting method: Schmidt')
## plt.grid(True)
## plt.show()
######################################################################
# VPA
## ENERGY
N_E = 1024
E_min = 1e-6 * meV
E_max = 60 * meV
E = np.linspace(E_min, E_max, N_E)
k = np.sqrt(2 * M_red * E) / hbar

delta_0 = 0 # TAYLOR IC

## solve ODEs (equations (31), (32) in OSU) for multiple k using VPA 
######################################################################
def ODE_32(r, delta, k, l):
    # Set l value manually different energy scattering
    top = -(2 * M_red * V_Xe(r) / (hbar**2)) * (np.cos(delta) * riccati_jn(l, k * r)[0][l] - np.sin(delta) * riccati_yn(l, k * r)[0][l])**2
    ddelta_dr = top / k 
    return ddelta_dr

# ODE solver ODEINT
delta = []
for i, k_i in enumerate(k):
    # print(i)
    l = 0
    solution_ODE = odeint(
                        lambda r, delta : ODE_32(r, delta, k_i, l),
                        delta_0, r, tfirst=True
                        )
    delta.append(solution_ODE[-1] - np.pi)

# print(f'\n{delta=}\n')

# A plot of delta vs E:
plt.plot(E / meV,  delta)
plt.axis(xmin=E_min/ meV, xmax=E_max / meV)
plt.xlabel(r'$E$ (meV) ')
plt.ylabel(r'$\delta^{s} (E)$')
plt.title(r'Energy dependent symmetric phase shifts ($\ell \neq 0$) ')
plt.grid(True)
plt.savefig('symmetric_phase_shifts.png')
plt.show()



#  A plot of e^(pi cot delta) vs E:
cot_delta = 1 / np.tan(delta)
plt.plot(E / meV, np.exp(np.pi * cot_delta))
plt.axis(ymin=0, ymax=0.1)
plt.axis(xmin=E_min/ meV, xmax=3)
plt.xlabel(r'$E$ (meV) ')
plt.ylabel(r'$e^{\pi cot(\delta)}$')
plt.title(r'$e^{\pi cot(\delta)}$ vs $E$')
plt.grid(True)
plt.savefig('symmetric_phase_shifts_exp_plot.png')
plt.show()

######################################################################

