"""
A restricted Hartree-Fock script using the Psi4NumPy Formalism
References:
- Algorithm taken from [Szabo:1996], pp. 146
- Equations taken from [Szabo:1996]
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-9-30"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Memory for Psi4 in GB
psi4.set_memory('500 MB')
psi4.core.set_output_file("output.dat", False)

# Memory for numpy in GB
numpy_memory = 2

mol = psi4.geometry("""
1 1 
H 0 0 0  
He 1.4632 0 0 
symmetry c1
units bohr
""")

psi4.set_options({'basis': 'sto-3g',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8})

# Set defaults
maxiter = 40
E_conv = 1.0E-6
D_conv = 1.0E-3

# Integral generation from Psi4's MintsHelper
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))

mints = psi4.core.MintsHelper(wfn.basisset())
S = np.asarray(mints.ao_overlap())
V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())
nbf = S.shape[0]
ndocc = wfn.nalpha()
I = np.asarray(mints.ao_eri())

H = T + V

# Orthogonalizer A = S^(-1/2) using Psi4's matrix power.
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

# Calculate initial core guess: [Szabo:1996] pp. 145
Hp = A.dot(H).dot(A)            # Eqn. 3.177
e, C2 = np.linalg.eigh(Hp)      # Solving Eqn. 1.178
C = A.dot(C2)                   # Back transform, Eqn. 3.174
Cocc = C[:, :ndocc]

D = np.einsum('pi,qi->pq', Cocc, Cocc) # [Szabo:1996] Eqn. 3.145, pp. 139


print('\nStart SCF iterations:\n')
t = time.time()
E = 0.0
Enuc = mol.nuclear_repulsion_energy()
Eold = 0.0
Dold = np.zeros_like(D)

E_1el = np.einsum('pq,pq->', H + H, D) + Enuc
print('One-electron energy = %4.16f' % E_1el)

for SCF_ITER in range(1, maxiter + 1):

    # Build fock matrix: [Szabo:1996] Eqn. 3.154, pp. 141
    J = np.einsum('pqrs,rs->pq', I, D)
    K = np.einsum('prqs,rs->pq', I, D)
    F = H + J * 2 - K

    diis_e = np.einsum('ij,jk,kl->il', F, D, S) - np.einsum('ij,jk,kl->il', S, D, F)
    diis_e = A.dot(diis_e).dot(A)
    dRMS = np.mean(diis_e**2)**0.5

    # SCF energy and update: [Szabo:1996], Eqn. 3.184, pp. 150
    SCF_E = np.einsum('pq,pq->', F + H, D) + Enuc

    print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E' % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
    if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
        break

    Eold = SCF_E
    Dold = D

    # Diagonalize Fock matrix: [Szabo:1996] pp. 145
    Fp = A.dot(F).dot(A)                   # Eqn. 3.177
    e, C2 = np.linalg.eigh(Fp)             # Solving Eqn. 1.178
    C = A.dot(C2)                          # Back transform, Eqn. 3.174
    Cocc = C[:, :ndocc]
    D = np.einsum('pi,qi->pq', Cocc, Cocc) # [Szabo:1996] Eqn. 3.145, pp. 139

    if SCF_ITER == maxiter:
        psi4.core.clean()
        raise Exception("Maximum number of SCF cycles exceeded.")


print('Final SCF energy: %.8f hartree' % SCF_E)
SCF_E_psi = psi4.energy('SCF')
psi4.compare_values(SCF_E_psi, SCF_E, 6, 'SCF Energy')
print(SCF_E_psi)
print(SCF_E)
