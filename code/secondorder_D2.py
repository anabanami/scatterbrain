from __future__ import division, print_function
from scipy.special import j0, jn_zeros
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

# Parameters
r_max = 1.0
dr = 0.01
r = np.arange(0, r_max, dr)
npts = r.shape[0]
nvals = 10

# Matrices
D2 = diags([1, -2, 1], [-1, 0, 1], shape=(npts - 1, npts - 1))/dr**2
D1 = diags([-0.5, 0, 0.5], [-1, 0, 1], shape=(npts - 1, npts - 1))/dr
r_new = r[0:npts-1] + dr/2
R_inv = diags(1/r_new, 0)
A = D2 + R_inv.dot(D1)

# Finite differences
vals, vecs = eigs(-A, k=nvals, which="SM") 
vecs = np.vstack((vecs, np.zeros(vecs.shape[1])))

# Analytic results
vals_anal = jn_zeros(0, nvals)
vecs_anal = j0(np.outer(r, vals_anal)/r_max)


## Plots

# Eigenvalues
plt.figure()
plt.plot(vals_anal**2)
plt.plot(vals, "ok")
plt.legend(["Analytic eigenvalues",
            "Numeric eigenvalues"])
plt.xlabel(r"$n$")
plt.ylabel(r"$E_n$")


# Eigenvectors
plt.figure()
plt.subplot(121)
plt.plot(r, np.abs(vecs[:, 0:4]/vecs[0, 0:4])**2)
plt.title("Finite differences")
plt.xlabel(r"$r$")
plt.ylabel(r"$|\psi|^2$")

plt.subplot(122)
plt.plot(r, np.abs(vecs_anal[:, 0:4])**2)
plt.title("Analytic solution")
plt.xlabel(r"$r$")
plt.ylabel(r"$|\psi|^2$")

plt.tight_layout()
plt.show()
