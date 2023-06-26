#!/usr/bin/env python 
import numpy as np
from scf_functions import *

def main():
    max_iter = 50
    E_conv = 1e-7
    P_conv = 1e-3

    #Define coordinates in array
    coords = np.asarray([[0.000, 0.000, 0.000], [1.4632, 0.000, 0.000]])
    distances = shift_distances(coords)
    
    #Define elements
    elems = ['He', 'H']
    #For now, store basis set for STO-3G in a dictionary
    #basis_dict_sto3g = {'H': [[[0.444635, 0.168856], [0.535328, 0.623913], [0.154329, 3.42525]]], 'He': [[[0.44463454, 0.31364979], [0.535328, 1.15892300], [0.15432897, 6.36242139]]]}
    basis_dict_sto3g = {'H': [[[0.444635, 0.168856], [0.535328, 0.623913], [0.154329, 3.42525]]], 'He': [[[0.444635, 0.480844], [0.535328, 1.77669], [0.154329, 9.75393]]]}
    basis_dict_631g = {'H': [[[0.0334946, 18.7311370], [0.23472695, 2.8253937], [0.81375733, 0.6401217]], [1.0, 0.1612778]], 'He': [[[0.0237660, 38.421634], [0.1546790, 5.7780300], [0.4696300, 1.2417740]], [1.0, 0.2979640]]}
    #Nucleus charge
    charge_dict = {'H': 1.0, 'He': 2.0}

    #Get basis and charge for each atom
    basis = []
    charge = []
    nuc_charge = 0
    elem = 0
    for i in elems:
        for func in basis_dict_sto3g[i]:
            basis.append(Basis_Function(func, distances[elem]))
        charge.append(charge_dict[i])
        nuc_charge = nuc_charge + charge[-1]
        elem += 1
    
    #Determines the number of electrons in the molecule
    total_charge = 1.0
    N = int(nuc_charge - total_charge)
    n_orb = int(N/2)

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
