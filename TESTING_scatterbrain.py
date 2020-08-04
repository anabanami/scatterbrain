from scipy.integrate import odeint
from scipy.integrate import solve_ivp
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

# TEST potentials
def square_well(r, V_0):
    # Square well potential with adjustable depth V_0
    V_sw = V_0 * np.heaviside(r_star - r, 0)
    return V_sw

def Coulomb(r):
    # Coulomb potential with width equivalent to Schmidt potential
    Q = -3.5456393493840546e-39
    V_E = (1 / (4 * np.pi * epsilon) * (Q / r))
    return V_E

#######################Bound states search############################
##########################DIAGONALISATION#############################
## DIRECT DIAGONALISATION ZERO BCS
## BOUND STATES serch. Setting up our matrices:
# A = np.zeros((N,N)) # main matrix
# dr = r[1] - r[0] # step size

## ZERO BC r->0 , free BC r->Rmax (NON-HERMITIAN)
# second_D = np.zeros_like(A)
## second derivative matrix (4th order approximation)
# for j in range(N):
    # if j == 0:
    #     second_D[j, j] = -5 / 2
    #     second_D[j, j+1] = 4 / 3
    #     second_D[j, j+2] = -1 / 12
    # elif j == 1:
    #     second_D[j, j-1] = 4 / 3
    #     second_D[j, j] = -5 / 2
    #     second_D[j, j+1] = 4 / 3
    #     second_D[j, j+2] = -1 / 12
    # elif 2 <= j <= N-3:
    #     second_D[j, j-2] = -1 / 12
    #     second_D[j, j-1] = 4 / 3
    #     second_D[j, j] = -5 / 2
    #     second_D[j, j+1] = 4 / 3
    #     second_D[j, j+2] = -1 / 12
    # elif j == N-2:
    #     second_D[j, j+1] =  11 / 12
    #     second_D[j, j] = -20 / 12
    #     second_D[j, j-1] = 6 / 12
    #     second_D[j, j-2] = 4 / 12
    #     second_D[j, j-3] = -1 / 12
    # elif j == N - 1:
    #     second_D[j, j] = 35 / 12
    #     second_D[j, j-1] = -104 / 12
    #     second_D[j, j-2] = 114 / 12
    #     second_D[j, j-3] = -56 / 12
    #     second_D[j, j-4] = 11 / 12
    # else:
    #     raise ValueError('forgot a j')

## ZERO BCS (HERMITIAN MATRIX)
## second derivative matrix (4th order approximation)
    # if j == 0:
    #     second_D[j, j] = -5 / 2
    #     second_D[j, j+1] = 4 / 3
    #     second_D[j, j+2] = -1 / 12
    # elif j == 1:
    #     second_D[j, j-1] = 4 / 3
    #     second_D[j, j] = -5 / 2
    #     second_D[j, j+1] = 4 / 3
    #     second_D[j, j+2] = -1 / 12
    # elif 2 <= j <= N-3:
    #     second_D[j, j-2] = -1 / 12
    #     second_D[j, j-1] = 4 / 3
    #     second_D[j, j] = -5 / 2
    #     second_D[j, j+1] = 4 / 3
    #     second_D[j, j+2] = -1 / 12
    # elif j == N-2:
    #     second_D[j, j-2] = -1 / 12
    #     second_D[j, j-1] = 4 / 3
    #     second_D[j, j] = -5 / 2
    #     second_D[j, j+1] = 4 / 3
    # elif j == N - 1:
    #     second_D[j, j-2] = -1 / 12
    #     second_D[j, j-1] = 4 / 3
    #     second_D[j, j] = -5 / 2
    # else:
    #     raise ValueError('forgot a j')
# print('second_D:\n')
# print(second_D)

# Schmidt_matrix_eff = np.zeros_like(A)
# for j in range(N):
#     Schmidt_matrix_eff[j, j] = (1 / (4 * r[j]**2)) - (2 * M_red / hbar**2) * V_Xe(r[j])
# # print('this is Schmidt_matrix')
# # print(Schmidt_matrix_eff)

# A = (1 / dr**2) * second_D + Schmidt_matrix_eff # change of variables
## Diagonalise A
# eigenvalues, unitary = np.linalg.eig(A) # unitary's columns are eigenvectors of A
## eigenvalues, unitary


## eigenENERGIES
# E = -(eigenvalues * hbar**2) / (2 * M_red)
## PLOTTING BOUND EIGENSTATES
# for i, E_i in enumerate(E):
#     if not v_0 <= E_i <= 0:
#         continue
#     # PLOT Schmidt potential and states
#     plt.axhline(E_i /  meV, linestyle=":")
#     plt.plot(
#         r / angstrom,
#         - np.sign(unitary[1,i] - unitary[0,i]) * v_0 * unitary[:, i] / (abs(unitary[:, i]).max() * 5  * meV) + E_i / meV, 
#         label='$u_0(r)$'
#     )
#     print(f"eigenenergy {i}: {E_i / meV:+.02f} meV")

# plt.legend(loc="upper right")
# plt.axis(xmin=0)
# plt.plot(r / angstrom, V_Xe(r) / meV)
# plt.ylabel(r'$V_Xe$ (meV) and $u_{i}$ (arb) ')
# plt.xlabel(r'$r$ (Å)')
# plt.grid(True)
# plt.savefig('Schmidt_potential_and_groundstate_zero_BCS.png')
# plt.show()
##########################DIAGONALISATION#############################
############################SOLVE_IVP#################################
# # Bound states search: SOLVE_IVP
# # SHOOTING method

# ICS
# y_0 = [np.sqrt(r[0]), 1 / (2 * np.sqrt(r[0]))]

# # INITIAL ENERGY BOUNDS
# n = 100
# E_lower = v_0
# E_upper = 0
# E = np.linspace(E_lower, E_upper, n)

# # ODE
# def ODE_shooting(r, y, E):
#     u, w = y
#     result = np.zeros_like(y)
#     if abs(u) > 1e100: # if blowing up return zero derivative
#         return result
#     else:
#         result[0] = w
#         result[1] = (- 1 / (4 * r**2) +  2 * M_red * (V_Xe(r) - E) / hbar**2) * u
#         return result

# signs = []
# Energies = []
# for i, E_i in enumerate(E):
#     solution_ODE_shooting = solve_ivp(
#                                     lambda r, y: ODE_shooting(r, y, E_i),
#                                     [r[0], r[-1]], y_0, t_eval=r
#                                      )

#     u = solution_ODE_shooting.y[0]
#     # print(f"{u=}")
#     signs.append(np.sign(u[-1])) # storing the energy SIGN of solutions at Rmax
#     Energies.append(E_i)

# # print(f"\n{Energies=}\n")
# # print(f"\n{signs=}\n")

# E_flip_signs = []
# E_flip = []
# for i in range(len(signs) - 1):
#     if not signs[i] == signs[i+1]:
#         E_flip.append([Energies[i], Energies[i+1]])
#         E_flip_signs.append([signs[i], signs[i+1]])
#     else:
#         continue

# # print(f"\n{E_flip=}\n")
# # print(f"\n{E_flip_signs=}\n")

# for i in range(len(E_flip)):
#     E_lower, E_upper = E_flip[i]
#     sign_lower, sign_upper = E_flip_signs[i]

#     print("Hello, this is the start of the while loop")
#     while abs(E_upper - E_lower) > 1e-12 * meV:
#         E_mid = (E_lower + E_upper) / 2
#         # print(f"{E_mid / meV=}")
#         solution_ODE_shooting = solve_ivp(
#                                         lambda r, y: ODE_shooting(r, y, E_mid),
#                                         [r[0], r[-1]], y_0, t_eval=r
#                                          )
#         w = solution_ODE_shooting.y[0]

#         # plt.plot(r / angstrom, w)

#         if np.sign(w[-1]) == sign_lower:
#             # print("if statement")
#             E_lower = E_mid
#         else:
#             # print("else statement")
#             E_upper = E_mid

#     print("Hello, this is the end of the while loop")


# plt.plot(r / angstrom, 10 * w / max(abs(w[r < r_star])) + E_mid / meV)
# plt.plot(r / angstrom, V_Xe(r) / meV)
# plt.axhline(E_mid / meV, linestyle=":")
# plt.axis(xmin=0, xmax=400)
# plt.figtext(0.65,0.15, f"$E_0 = {E_mid / meV:.03f} $ meV")
# plt.ylabel(r'$V_Xe$ (meV) and $u_{i}$ (arb) ')
# plt.xlabel(r'$r$ (Å)')
# plt.title('Shooting method: Schmidt bound state')
# plt.savefig('schmidt_bound_state.png')
# plt.show()
#############################SOLVE_IVP################################
#######################Bound states search############################

##############################VPA#####################################
#############################SOLVE_IVP################################
## VPA
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

# def ODE_32_square(r, delta, k, l):
#     # Set l value manually different energy scattering
#     top = -(2 * M_red * square_well(r, v_0) / (hbar**2)) * (np.cos(delta) * riccati_jn(l, k * r)[0][l] - np.sin(delta) * riccati_yn(l, k * r)[0][l])**2
#     ddelta_dr = top / k 
    return ddelta_dr

# ODE solver: ODEINT
delta = []
for i, k_i in enumerate(k):
    # print(i)
    l = 0     # Set l value manually different energy scattering
    solution_ODE = odeint(
                        lambda r, delta : ODE_32(r, delta, k_i, l),
                        delta_0, r, tfirst=True
                        )
    delta.append(solution_ODE[-1] - np.pi)

# print(f'\n{delta=}\n')

# A plot of delta vs E:

# plt.plot(E / meV,  delta, label=r'$\delta^{n}(E)$')

# plt.axis(xmin=E_min / meV, xmax=E_max / meV, ymin=-3.0, ymax=0)
# plt.xlabel(r'$E$ (meV) ')
# plt.ylabel(r'$\delta^{s} (E)$'))
# plt.title(r'Energy dependent symmetric phase shifts ($\ell \neq 0$)')
# plt.grid(True)
# plt.savefig('symmetric_phase_shifts.png')
# plt.show()
#########################ANALYTIC SOLUTION############################
## TESTING VPA'S RESULTS
## parameters are the same as NUMERICS ABOVE

# delta_analytic = k * (np.sqrt(1 - 2 * M_red * v_0 / (hbar**2 * k**2)) - 1) * r_star #LOGIC BASED ON PLANE WAVE PHASE
delta_OSU = np.arctan(
    np.sqrt(k**2 / (k**2 - 2 * M_red * v_0 / hbar**2))
    * np.tan(r_star * np.sqrt(k**2 - 2 * M_red * v_0 / hbar**2))
    ) - r_star * k # OSU
# delta_analytic = np.arctan(np.tan(delta_OSU))
delta_analytic = k / np.tan(delta_OSU) # discontinuity fix OSU

print(f"{delta_analytic=}")

# A plot of delta_analytic vs E:
plt.plot(E / meV,  delta_analytic - np.pi, label=r'$\delta^{a}(E)$')
# plt.axis(xmin=E_min/ meV, xmax=E_max / meV, ymin=-3.0, ymax=0)
plt.xlabel(r'$E$ (meV) ')
plt.ylabel(r'$\delta^{s} (E)$')
plt.title(r'Analytic and numeric symmetric phase shifts')
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('analytic_numeric_symmetric_phase_shifts_vs_energy.png')


plt.show()
########################ANALYTIC SOLUTION############################

# # A plot of e^(pi cot delta) vs E:
# cot_delta = 1 / np.tan(delta)
# plt.plot(E / meV, np.exp(np.pi * cot_delta))
# plt.axis(xmin=E_min/ meV, xmax=3)
# plt.xlabel(r'$E$ (meV) ')
# plt.ylabel(r'$e^{\pi cot(\delta)}$')
# plt.title(r'$e^{\pi cot(\delta)}$ vs $E$')
# plt.grid(True)
# plt.savefig('exp_plot_symmetric_phase_shifts.png')
# plt.show()
#############################SOLVE_IVP################################
##############################VPA#####################################


############################other tests###############################
# Potentials
# PLOTS

# Schmidt potential
# plt.plot(r / angstrom, V_Xe(r) / meV)
# ylabel = plt.ylabel(r'$V_{Xe}$ (meV)', fontsize=10)
# plt.xlabel(r'$r$ (Å)')
# plt.grid(True)
# plt.title('Schmidt potential')
# plt.axis(xmin=0)
# plt.savefig('V_Xe_Schmidt.png')
# plt.show()

# # Square well potential
# plt.plot(r / angstrom, square_well(r, v_0) / meV)
# ylabel = plt.ylabel(r'$V_{sw}(r)$ (meV)', fontsize=10)
# plt.xlabel(r'$r$ (Å)')
# plt.savefig('V_sw.png')
# plt.axis(xmin=0)
# plt.grid(True)
# plt.title('Square well potential')
# plt.show()

# Coulomb potential
# plt.plot(r / angstrom, Coulomb(r) / meV)
# plt.xlim([0, 200])
# plt.ylim([-1e3, 1e2])
# ylabel = plt.ylabel(r'$V_{E}$ (meV)', fontsize=10)
# plt.xlabel(r'$r$ (Å)')
# plt.savefig('V_Coulomb.png')
# plt.grid(True)
# plt.title('Coulomb potential')
# plt.show()
#####################COULOMB POTENTIAL###TEST#####################
# # Coulomb matrix
# V_E = np.zeros_like(A)
# for j in range(N):
#     V_E[j, j] = (1 / (4 * r[j]**2)) - (2 * M_red / hbar**2) * Coulomb(r[j])

# A = (1 / dr**2) * second_D + V_E

# # Diagonalise A
# eigenvalues, unitary = np.linalg.eig(A) # unitary's columns are eigenvectors of A

# E = -(eigenvalues * hbar**2) / (2 * M_red)
# for i, E_i in enumerate(E):
#     if not -3e2 <= E_i <= 0:
#         continue
#     plt.axhline(E_i /  meV, linestyle=":")
#     plt.plot(
#         r / angstrom,
#         - np.sign(unitary[1,i] - unitary[0,i]) * v_0 * unitary[:, i] / (abs(unitary[:, i]).max() * 5  * meV) + E_i / meV, 
#         label=f'$u_{{{i}}}(r)$'
#     )
#     print(f"eigenenergy {i}: {E_i / meV:+.02f} meV")


# plt.legend(loc="lower right")
# plt.xlim([0, 200])
# plt.ylim([-3e2, 1e2])
# plt.plot(r / angstrom, Coulomb(r) / meV)
# ylabel = plt.ylabel(r'$V_E$ (meV) and $u_{i}$ (arb) ')
# plt.xlabel(r'$r$ (Å)')
# plt.title('Coulomb bound states')
# plt.grid(True)
# plt.savefig('Coulomb_bound_states.png')
# plt.show()
#####################COULOMB POTENTIAL###TEST#####################
########################other tests###############################