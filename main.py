#!/usr/bin/env python 
import numpy as np
from scf_functions import *

def main():
    #Define coordinates in array
    coords = np.asarray([[0.000, 0.000, 0.000], [1.4, 0.000, 0.000]])
    distances = shift_distances(coords)

    #Define elements
    elems = ['H', 'H']
    #For now, store basis set for STO-3G in a dictionary
    basis_dict = {'H': [[0.444635, 0.168856], [0.535328, 0.623913], [0.154329, 3.42525]], 'He': [[0.444635, 0.480844], [0.535328, 1.77669], [0.154329, 9.75393]]}
    #Nucleus charge
    charge_dict = {'H': 1.0, 'He': 2.0}

    #Get basis and charge for each atom
    basis = []
    charge = []
    nuc_charge = 0
    for i in elems:
        basis.append(Basis_Function(basis_dict[i]))
        charge.append(charge_dict[i])
        nuc_charge = nuc_charge + charge[-1]

    #Determines the number of electrons in the molecule
    total_charge = 0.0
    N = int(nuc_charge - total_charge)
    
    #Define S, T, V and ERI matrices
    s_matrix = np.zeros((len(elems), len(elems)))
    t_matrix = np.zeros((len(elems), len(elems)))
    v_matrix = np.zeros((len(elems), len(elems)))
    twoelec_matrix = np.zeros((len(elems), len(elems), len(elems), len(elems)))

    #Computes one-electron integrals
    for i in range(len(basis)):
        for j in range(len(basis)):
            for k in range(basis[i].num_priv):
                for l in range(basis[j].num_priv):
                    overlap = overlap_integral(basis[i].coeff[k], basis[j].coeff[l], basis[i].alpha[k], basis[j].alpha[l], distances[i], distances[j])
                    s_matrix[i, j] += overlap
                    kinetic = kinetic_integral(basis[i].coeff[k], basis[j].coeff[l], basis[i].alpha[k], basis[j].alpha[l], distances[i], distances[j])
                    t_matrix[i, j] += kinetic
                    nuc_attrac = nucattrac_integral(basis[i].coeff[k], basis[j].coeff[l], basis[i].alpha[k], basis[j].alpha[l], distances[i], distances[j], distances, charge)
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
                                    two_elec = two_electron(basis[i].coeff[m], basis[j].coeff[n], basis[k].coeff[o], basis[l].coeff[p], basis[i].alpha[m], basis[j].alpha[n], basis[k].alpha[o], basis[l].alpha[p], distances[i], distances[j], distances[k], distances[l])
                                    twoelec_matrix[i, j, k, l] += two_elec

    #Orthogonalize the S matrix
    X = orthogonalize(s_matrix, canonical=True)

    #Initial guess for Fock matrix is H_core for now
    F = H_core
    #Don't have DIIS implemented, so loop 50 times for SCF
    for i in range(50):
        #Orthogonalize F to obtain F_prime
        F_prime = np.matmul(np.matmul(X.transpose(), F), X)

        #Get eigenvalues (eig) and eigenvectors (C_prime) from F_prime
        eig, C_prime = np.linalg.eig(F_prime)

        #Get C from C_prime
        C = np.matmul(X, C_prime)

        #New density matrix
        P = density_matrix(C, num_elecs=N)

        #Form new two electron integral from density matrix
        G = g_matrix(P, twoelec_matrix)

        #New Fock matrix
        F = H_core + G

        #Compute current energy
        energy = get_energy(P, H_core, F)

    #Get nuclear repulsion energy
    nuclear_repul = get_nuclear(distances, charge)

    #Total energy
    Etot = energy + nuclear_repul
    print(Etot)

if __name__ == "__main__":
    main()
