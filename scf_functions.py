#!/usr/bin/env python
import numpy as np
from scipy.special import erf
from scipy.linalg import fractional_matrix_power

def shift_distances(coords):
    """
    Shift first atom to the origin and compute distances to other atoms
    from the first atom
    """
    origin = np.asarray([0, 0, 0])

    #Shift 1st element to be at origin
    offsets = coords[0] - origin
    coords -= offsets

    coords -= coords[0]
    dists = np.sqrt((coords**2).sum(axis=1))

    return dists

def F(t):
    """
    Compute F function, which is related to error function and
    used in electron-nuclear attraction and two-electron integrals
    """
    if t < 1.0e-6: return 1.0
    else: return 0.5 * (np.pi/t)**(0.5) * erf(t**(0.5))

def overlap_integral(d1, d2, alpha1, alpha2, RA, RB):
    """
    Compute overlap integral between two basis functions
    """
    coeff = d1*d2
    norm = ((4*alpha1*alpha2)/np.pi**2)**(3/4)
    term = norm * (np.pi/(alpha1+alpha2))**(3/2) * np.exp(-alpha1*alpha2/(alpha1+alpha2) * (RA - RB)**2)

    return coeff * term

def kinetic_integral(d1, d2, alpha1, alpha2, RA, RB):
    """
    Compute kinetic energy integral
    """
    coeff = d1*d2
    norm = ((4*alpha1*alpha2)/np.pi**2)**(3/4)
    term = alpha1*alpha2/(alpha1+alpha2) * (3.0 - 2.0*alpha1*alpha2/(alpha1+alpha2)*(RA-RB)**2) * norm * (np.pi/(alpha1+alpha2))**(3/2) * np.exp(-alpha1*alpha2/(alpha1+alpha2) * (RA - RB)**2)

    return coeff * term

def nucattrac_integral(d1, d2, alpha1, alpha2, RA, RB, R, charge):
    """
    Compute electron-nuclear attraction integral
    """
    coeff = d1*d2
    norm = ((4*alpha1*alpha2)/np.pi**2)**(3/4)
    v = 0
    for i in range(len(R)):
        K = -2 * np.pi / (alpha1+alpha2) * charge[i] * np.exp(-alpha1*alpha2/(alpha1+alpha2) * (RA - RB)**2)
        Rp = (alpha1*RA + alpha2*RB)/(alpha1+alpha2)
        v += norm * K * F((alpha1+alpha2) * (Rp - R[i])**2)

    return coeff * v

def two_electron(d1, d2, d3, d4, alpha1, alpha2, alpha3, alpha4, RA, RB, RC, RD):
    """
    Compute the two-electron integral
    """
    coeff = d1*d2*d3*d4
    norm = ((16*alpha1*alpha2*alpha3*alpha4)/np.pi**4)**(3/4)
    term1 = 2*np.pi**(5/2)/((alpha1+alpha2)*(alpha3+alpha4)*(alpha1+alpha2+alpha3+alpha4)**(1/2))
    term2 = np.exp(-alpha1*alpha2/(alpha1+alpha2)*(RA-RB)**2 - alpha3*alpha4/(alpha3+alpha4)*(RC-RD)**2)
    Rp = (alpha1*RA + alpha2*RB)/(alpha1 + alpha2)
    Rq = (alpha3*RC + alpha4*RD)/(alpha3 + alpha4)
    term3 = F((alpha1+alpha2)*(alpha3+alpha4)/(alpha1+alpha2+alpha3+alpha4)*(Rp - Rq)**2)

    return coeff * norm * term1 * term2 * term3

def symmetric_orthogonalization(U, s):
    """
    Perform symmetric orthogonalization
    """
    return np.matmul(np.matmul(U, s), U.transpose())

def canonical_orthogonalization(U, s):
    """
    Perform canonical orthogonalization
    """
    return np.matmul(U, s)

def orthogonalize(S, min_tol=1e-7, canonical=False):
    """
    Performs either canonical or symmetric orthogonalization 
    of the overlap integral. Default is symmetric orthogonalization
    unless canonical=True or the minimum eigenvalue is less than min_tol
    """
    eig, U = np.linalg.eig(S)
    U = np.flipud(U)
    s = np.matmul(np.matmul(U.transpose(), S), U)
    s = fractional_matrix_power(s, -0.5)
    if eig.min() > min_tol and not canonical:
        return symmetric_orthogonalization(U, s)
    else:
        return canonical_orthogonalization(U, s)

def density_matrix(C, n_elec):
    """
    Computes the density matrix from
    coefficient matrix
    """
    P = np.zeros((n_elec, n_elec))
    for e in range(int(n_elec/2)):
        for i in range(C.shape[0]):
            for j in range(C.shape[0]):
                P[i, j] += C[i, e] * C[j, e]
    return 2.0*P

def g_matrix(P, twoelec):
    """
    Computes the G matrix with density matrix and two-electron integrals
    """
    G = np.zeros_like(P)

    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            for k in range(G.shape[0]):
                for l in range(G.shape[0]):
                    G[i, j] += P[k, l] * (twoelec[i, j, k, l] - 0.5 * twoelec[i, l, k, j])
    return G

def get_energy(P, H_core, F):
    """
    Get the current SCF energy
    """
    energy = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            energy += P[j, i] * (H_core[i, j] + F[i, j])
    return 0.5 * energy

def get_nuclear(distances, charge):
    """
    Get nuclear repulsion term
    """
    nuclear = 0
    for i in range(0, distances.shape[0]-1):
        for j in range(i+1, distances.shape[0]):
            nuclear += charge[i]*charge[j]/distances[j]
    return nuclear

def test_convergence(E, E_old, E_conv, P, P_old, P_conv):
    P_change = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            P_change += (P[i,j] - P_old[i,j])**2
    P_change = np.sqrt(P_change/P.shape[0]**2)
    if (np.abs(E - E_old) < E_conv) and (P_change < P_conv):
        return True
    else:
        return False

class Basis_Function:
    """
    class to store basis function info, currently works with
    STO-3G
    """
    def __init__(self, basis_def, center):
        self.coeff = []
        self.alpha = []
        self.center = center

        if isinstance(basis_def[0], list):
            self.num_priv = len(basis_def)
            for i in basis_def:
                self.coeff.append(i[0])
                self.alpha.append(i[1])
        else:
            self.num_priv = 1
            self.coeff.append(basis_def[0])
            self.alpha.append(basis_def[1])

