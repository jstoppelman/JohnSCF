#!/usr/bin/env python 
import numpy as np
from ase.io import read
from basis import Basis
from scf_functions import *
import sys

def main():
    max_iter = 50
    E_conv = 1e-7
    P_conv = 1e-3

    atoms = read(sys.argv[1])

    #Define elements
    elems = atoms.get_chemical_symbols()

    #Handle basis set information
    basis = Basis(atoms, 'sto-3g')
    
    #Determines the number of electrons in the molecule
    total_charge = basis.get_total_charge()
    nuclear_charge_sum = basis.get_nuclear_charge()
    #N is the number of electrons
    N = nuclear_charge_sum - total_charge
    sys.exit()

    #Define S, T, V and ERI matrices
    s_matrix = np.zeros((len(basis), len(basis)))
    t_matrix = np.zeros((len(basis), len(basis)))
    v_matrix = np.zeros((len(basis), len(basis)))
    twoelec_matrix = np.zeros((len(basis), len(basis), len(basis), len(basis)))

    #Computes one-electron integrals
    for i in range(len(basis)):
        for j in range(len(basis)):
            for k in range(basis[i].num_priv):
                for l in range(basis[j].num_priv):
                    overlap = overlap_integral(basis[i].coeff[k], basis[j].coeff[l], basis[i].alpha[k], basis[j].alpha[l], basis[i].center, basis[j].center)
                    s_matrix[i, j] += overlap

                    kinetic = kinetic_integral(basis[i].coeff[k], basis[j].coeff[l], basis[i].alpha[k], basis[j].alpha[l], basis[i].center, basis[j].center)
                    t_matrix[i, j] += kinetic

                    nuc_attrac = nucattrac_integral(basis[i].coeff[k], basis[j].coeff[l], basis[i].alpha[k], basis[j].alpha[l], basis[i].center, basis[j].center, distances, charge)
                    v_matrix[i, j] += nuc_attrac
   
    #Core Hamiltonian
    H_core = t_matrix + v_matrix
    
    #Compute two electron integrals, possibly recode so fewer for loops
    for i in range(len(basis)):
        for j in range(len(basis)):
            for k in range(len(basis)):
                for l in range(len(basis)):
                    for m in range(basis[i].num_priv):
                        for n in range(basis[j].num_priv):
                            for o in range(basis[k].num_priv):
                                for p in range(basis[l].num_priv):
                                    two_elec = two_electron(basis[i].coeff[m], basis[j].coeff[n], basis[k].coeff[o], basis[l].coeff[p], basis[i].alpha[m], basis[j].alpha[n], basis[k].alpha[o], basis[l].alpha[p], basis[i].center, basis[j].center, basis[k].center, basis[l].center)
                                    twoelec_matrix[i, j, k, l] += two_elec

    #Orthogonalize the S matrix
    X = orthogonalize(s_matrix, canonical=True)

    #Initial guess for Fock matrix is H_core for now
    F = H_core
    F_prime = np.matmul(np.matmul(X.transpose(), F), X)
    eig, C_prime = np.linalg.eig(F_prime)
    C = np.matmul(X, C_prime)
    C_orb = C[:, :n_orb]
    P = density_matrix(C_orb, N)
    print(P)
    sys.exit()
    E_old = 0.0
    P_old = np.zeros((len(basis), len(basis)))

    Fock_list = []
    DIIS_error = []

    for i in range(max_iter):
        print("Iteration", i)
        #Orthogonalize F to obtain F_prime
        F_prime = np.matmul(np.matmul(X.transpose(), F), X)

        #Get eigenvalues (eig) and eigenvectors (C_prime) from F_prime
        eig, C_prime = np.linalg.eig(F_prime)

        #Get C from C_prime
        C = np.matmul(X, C_prime)

        C_orb = C[:, :n_orb]

        #New density matrix
        P = density_matrix(C_orb, N)

        #Form new two electron integral from density matrix
        G = g_matrix(P, twoelec_matrix)

        #New Fock matrix
        F = H_core + G

        #Compute current energy
        E = get_energy(P, H_core, F)

        if test_convergence(E, E_old, E_conv, P, P_old, P_conv):
            break

        E_old = E
        P_old = P

    #Get nuclear repulsion energy
    nuclear_repul = get_nuclear(distances, charge)
    print(nuclear_repul)
    #Total energy
    Etot = E + nuclear_repul
    print(Etot)

if __name__ == "__main__":
    main()
