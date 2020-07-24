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
# adjustable parameters
min_physical_n = 0
N = 6 * 1024
r = np.linspace(1e-6, 200, N) * angstrom # r is smol
######################################################################

# ENERGY
E_min = 0.1 * meV
E_max = 60 * meV
E = np.linspace(E_min, E_max, N)
k = np.sqrt(2 * M_red * E) / hbar

# FUNCTIONS
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


######################################################################
# BOUND STATES serch. Setting up our matrices:
# def A_matrix(N, r):
A = np.zeros((N,N)) # main matrix
dr = r[1] - r[0] # step size

second_dR = np.zeros_like(A) # second derivative matrix (mixed? order approximation)
for j in range(N):
    if j == 0:
        second_dR[j, j] = 35 / 12
        second_dR[j, j+1] = -104 / 12
        second_dR[j, j+2] = 114 / 12
        second_dR[j, j+3] = -56 / 12
        second_dR[j, j+4] = 11 / 12
    elif j == 1:
        second_dR[j, j-1] =  11 / 12
        second_dR[j, j] = -20 / 12
        second_dR[j, j+1] = 6 / 12
        second_dR[j, j+2] = 4 / 12
        second_dR[j, j+3] = -1 / 12
    elif 2 < j < N-3:
        second_dR[j, j-2] = -1 / 12
        second_dR[j, j-1] = 4 / 3
        second_dR[j, j] = -5 / 2
        second_dR[j, j+1] = 4 / 3
        second_dR[j, j+2] = -1 / 12
    elif j == N-2:
        second_dR[j, j+1] =  11 / 12
        second_dR[j, j] = -20 / 12
        second_dR[j, j-1] = 6 / 12
        second_dR[j, j-2] = 4 / 12
        second_dR[j, j-3] = -1 / 12
    else:
        second_dR[j, j] = 35 / 12
        second_dR[j, j-1] = -104 / 12
        second_dR[j, j-2] = 114 / 12
        second_dR[j, j-3] = -56 / 12
        second_dR[j, j-4] = 11 / 12
# print('Second_dR:\n')
# print(second_dR)

Schmidt_matrix_eff = np.zeros_like(A)
for j in range(N):
    Schmidt_matrix_eff[j, j] = (1 / (4 * r[j]**2)) - (2 * M_red / hbar**2) * V_Xe(r[j])
# print('this is Schmidt_matrix')
# print(Schmidt_matrix_eff)

A = (1 / dr**2) * second_dR + Schmidt_matrix_eff # change of variables
# Diagonalise A
eigenvalues, unitary = np.linalg.eig(A) # unitary's columns are eigenvectors of A
eigenvalues, unitary
######################################################################


# TEST 
# print('The reduced mass:', M_red / mass_e)
# E_ground = -(eigenvalues[N-1-min_physical_n] * hbar**2) / (2 * M_red) 
# print('The ground state energy [meV]:', E_ground / meV)


# PLOTS

# Schmidt potential
# plt.plot(r / angstrom, V_Xe(r) / meV)
# ylabel = plt.ylabel(r'$V_{Xe}$ (meV)', fontsize=10)
# plt.xlabel(r'$r$ (Å)')
# plt.savefig('V_Xe_Schmidt.png')
# plt.axis(xmin=0)
# plt.show()

# Schmidt potential and states
for n in range(min_physical_n, 8):
    i = N - 1 - n
    E_i = -(eigenvalues[i] * hbar**2) / (2 * M_red)
    if n == 7:
        label = '$u_0(r)$'
    else:
        label = None
    plt.axhline(E_i /  meV, linestyle=":")
    plt.plot(
        r / angstrom,
        - np.sign(unitary[1,i] - unitary[0,i]) * v_0 * unitary[:, i] / (abs(unitary[:, i]).max() * 5  * meV) + E_i / meV, 
        label=label
    )
    print(f"eigenenergy {n - min_physical_n}: {E_i / meV:+.02f} meV")

plt.legend(loc="upper right")
plt.axis(xmin=0)
plt.plot(r / angstrom, V_Xe(r) / meV)
ylabel = plt.ylabel(r'$V_Xe$ (meV) and $R_{i}$ (arb) ')
plt.xlabel(r'$r$ (Å)')
plt.savefig('potential_and_states.png')
plt.show()