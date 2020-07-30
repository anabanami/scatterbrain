import matplotlib.pyplot as plt

eV = 1.602176634e-19 # [J]
meV = 1e-3 * eV

eigenvalues = [-27.04, -27.51, -27.89, -28.14, -28.24, -28.32, -28.38]
N = [1024 / 2, 1024, 1024 * 2, 1024 * 3, 1024 * 4, 1024 * 5, 1024 * 6]
plt.axhline(-31.7, linestyle=':',label='Schmidt value')
plt.plot(N, eigenvalues, 'bo-',label='numeric')
ylabel = plt.ylabel(r'$E$ (meV)')
plt.xlabel(r'$N$')
plt.legend()
plt.grid(True)
plt.savefig('ground_energy_vs_N.png')
plt.show()