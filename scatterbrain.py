from scipy.integrate import solve_ivp
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

#######################Bound states search############################
# SHOOTING method
# SOLVE_IVP
# ICS
y_0 = [np.sqrt(r[0]), 1 / (2 * np.sqrt(r[0]))]

# INITIAL ENERGY BOUNDS
n = 100
E_lower = v_0
E_upper = 0
E = np.linspace(E_lower, E_upper, n)

# ODE
def ODE_shooting(r, y, E):
    u, w = y
    result = np.zeros_like(y)
    if abs(u) > 1e100: # if blowing up return zero derivative
        return result
    else:
        result[0] = w
        result[1] = (- 1 / (4 * r**2) +  2 * M_red * (V_Xe(r) - E) / hbar**2) * u
        return result

signs = []
Energies = []
for i, E_i in enumerate(E):
    solution_ODE_shooting = solve_ivp(
                                    lambda r, y: ODE_shooting(r, y, E_i),
                                    [r[0], r[-1]], y_0, t_eval=r
                                     )
    u = solution_ODE_shooting.y[0]
    signs.append(np.sign(u[-1])) # storing the energy SIGN of solutions at Rmax
    Energies.append(E_i)

# print(f"\n{Energies=}\n")
# print(f"\n{signs=}\n")

E_flip_signs = []
E_flip = []
for i in range(len(signs) - 1):
    if not signs[i] == signs[i+1]:
        E_flip.append([Energies[i], Energies[i+1]])
        E_flip_signs.append([signs[i], signs[i+1]])
    else:
        continue

# print(f"\n{E_flip=}\n")
# print(f"\n{E_flip_signs=}\n")

for i in range(len(E_flip)):
    E_lower, E_upper = E_flip[i]
    sign_lower, sign_upper = E_flip_signs[i]

    print("Hello, this is the start of the while loop")
    while abs(E_upper - E_lower) > 1e-12 * meV:
        E_mid = (E_lower + E_upper) / 2
        # print(f"{E_mid / meV=}")
        solution_ODE_shooting = solve_ivp(
                                        lambda r, y: ODE_shooting(r, y, E_mid),
                                        [r[0], r[-1]], y_0, t_eval=r
                                         )
        w = solution_ODE_shooting.y[0]

        # plt.plot(r / angstrom, w)

        if np.sign(w[-1]) == sign_lower:
            # print("if statement")
            E_lower = E_mid
        else:
            # print("else statement")
            E_upper = E_mid

    print("Hello, this is the end of the while loop")


plt.plot(r / angstrom, 10 * w / max(abs(w[r < r_star])) + E_mid / meV)
plt.plot(r / angstrom, V_Xe(r) / meV)
plt.axhline(E_mid / meV, linestyle=":")
plt.axis(xmin=0, xmax=400)
plt.figtext(0.65,0.15, f"$E_0 = {E_mid / meV:.03f} $ meV")
plt.ylabel(r'$V_Xe$ (meV) and $u_{i}$ (arb) ')
plt.xlabel(r'$r$ (Å)')
plt.title('Shooting method: Schmidt bound state')
plt.savefig('schmidt_bound_state.png')
plt.show()
#######################Bound states search############################

##############################VPA#####################################
# VPA
# ENERGY
N_E = 1024
E_min = 1e-6 * meV
E_max = 60 * meV
E = np.linspace(E_min, E_max, N_E)
k = np.sqrt(2 * M_red * E) / hbar

delta_0 = 0 # TAYLOR IC

# solve ODEs (equations (31), (32) in OSU) for multiple k using VPA 

def ODE_31(r, delta, k):
    # l = 0
    ddelta_dr = -((2 * M_red * V_Xe(r) / hbar**2) * np.sin(k * r + delta)**2) / k 
    return ddelta_dr

# print(f'{ODE_31(angstrom, 0, 7e8)=}')

def ODE_32(r, delta, k, l):
    # Set l value manually different energy scattering
    top = -(2 * M_red * V_Xe(r) / (hbar**2)) * (np.cos(delta) * riccati_jn(l, k * r)[0][l] - np.sin(delta) * riccati_yn(l, k * r)[0][l])**2
    ddelta_dr = top / k 
    return ddelta_dr

# ODE solver: ODEINT
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
##############################VPA#####################################
