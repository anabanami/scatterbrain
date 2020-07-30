from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.special import struve, yn, riccati_jn, riccati_yn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 120

# GLOBALS
# physical constants and SI units
hbar = 1.0545718e-34 # [Js]
epsilon = 8.854e-12 
e_charge = 1.60217662e-19 # [C]
au = 1.64877727436e-41 # [C^2 m^2 J^(-1)] 
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
# DIAGONALISATION adjustable parameters
N =  50
r = np.linspace(1e-6, 200, N) * angstrom
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
    V_Xe[r < -r_star] = - (alpha * e_charge **2 / 2) * ((keldysh(abs(r)[r < -r_star] + dr) - keldysh(abs(r)[r < -r_star])) / dr)**2
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


######################################################################
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
######################################################################


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


######################################################################
## VPA
## ENERGY

E_min = 0.1 * meV
E_max = 60 * meV
E = np.linspace(E_min, E_max, N)
k = np.sqrt(2 * M_red * E) / hbar

delta_0 = [0] # TAYLOR IC

## solve ODEs (equations (31), (32) in OSU) for multiple k using VPA 
######################################################################
def ODE_31(r, delta, k, V_0):
    # l = 0
    top = (2 * M_red * V_Xe(r) / hbar**2) * np.sin(k * r + delta)**2
    ddelta_dr = - top / k 
    return ddelta_dr

def ODE_32(r, delta, k, V_0):
    #  l != 0
    top = (2 * M_red * V_Xe(r)) * (np.cos(delta) * riccati_jn(0, k * r)[0] - np.sin(delta) * riccati_yn(0, k * r)[0])**2
    ddelta_dr = - top / (hbar**2 * k) 
    return ddelta_dr
######################################################################
# ODE solver ODEINT
delta = []
for i, k_i in enumerate(k):
    # print(i)
    solution_ODE_31 = odeint(
                                lambda r, delta : ODE_31(r, delta, k_i, v_0),
                                 delta_0, r, tfirst=True
                            )
    delta.append(solution_ODE_31[1])

# print(f'\n{E=}\n')
# print(f'\n{np.shape(E)=}\n')
print(f'\nTHIS IS ODEINT {delta=}\n')
print(f'{np.shape(delta)=}\n')
# print(f'\n{solution_ODE_31=}\n')
# print(f'{np.shape(solution_ODE_31)=}\n')

plt.plot(E / meV,  delta)

plt.axis(xmin=E_min/ meV, xmax=E_max / meV)
plt.xlabel(r'$E$ (meV) ')
plt.ylabel(r'$\delta^{s} (E)$')
plt.title(r'Energy dependent symmetric phase shifts ($\ell = 0$) ')
plt.grid(True)
# plt.savefig('symmetric_phase_shifts.png')
plt.show()

########################SOLVE_IVP#####################################
# ODE solver SOLVE_IVP
# delta = []
# for i, k_i in enumerate(k):
#     # print(i)
#     solution_ODE_31 = solve_ivp(
#                                 lambda r, delta : ODE_31(r, delta, k_i, v_0),
#                                  [r[0], r[-1]], delta_0, t_eval=r
#                                  )
#     delta.append(solution_ODE_31.y[0, -1])

# print(f'THIS IS SOLVE_IVP {delta=}\n ')
# print(f'{np.shape(delta)=}\n')

# plt.plot(E / meV,  delta)
# plt.axis(xmin=E_min/ meV, xmax=E_max / meV)
# plt.xlabel(r'$E$ (meV) ')
# plt.ylabel(r'$\delta^{s} (E)$')
# plt.title('Energy dependent symmetric phase shifts ')
# plt.grid(True)
# # plt.savefig('symmetric_phase_shifts.png')
# plt.show()

######################################################################


######################################################################
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
######################################################################

###########################COULOMB POTENTIAL###TEST###################
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
###########################COULOMB POTENTIAL###TEST######################