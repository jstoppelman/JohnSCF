import numpy as np

class RHF:
    """
    Perform Restricted Hartree-Fock calculation
    """
    def __init__(self, overlap, 
            kinetic, 
            potential, 
            eri, 
            nuclear_repulsion, 
            basis, 
            maxiter=50, 
            E_conv=1e-8, 
            D_conv=1e-4):

        """
        Parameters
        -----------
        overlap: object
            Object for computing overlap integral
        kinetic: object
            Object for computing kinetic energy integral
        potential: object
            Object for computing nuclear-electron attraction
        eri: object
            Object for computing ERI
        nuclear_repulsion: object
            Object for computing nuclear-nuclear attractions
        basis: object
            Wrapper around Psi4 for reading basis data
        maxiter: int
            Number of SCF iterations
        E_conv: float
            SCF energy tolerance
        D_conv: float
            Convergence for density matrix
        """
        self.overlap = overlap
        self.kinetic = kinetic
        self.potential = potential
        self.eri = eri
        
        self.nuclear_repulsion = nuclear_repulsion
        self.basis = basis
        self.maxiter = maxiter
        self.E_conv = E_conv
        self.D_conv = D_conv

    def calculate_energy(self):
        """
        Calculate SCF energy, similar to Psi4Numpy
        """

        self.S = self.overlap.calculate()
        self.T = self.kinetic.calculate()
        self.V = self.potential.calculate()
        self.I = self.eri.calculate()

        self.H_core = self.T + self.V
        
        eig, eigv = np.linalg.eigh(self.S)
        s = np.diag(np.power(eig, -0.5))
        X = eigv @ s @ eigv.T
        
        H_core_p = X.T @ self.H_core @ X
        eig, C_p = np.linalg.eigh(H_core_p)
        C = X @ C_p
        
        occ = self.basis.wfn.nalpha()
        C_occ = C[:, :occ]

        #Form density matrix (alpha density matrix actually)
        P = C_occ @ C_occ.T

        E_nuc = self.nuclear_repulsion.calculate_nuclear_repulsion()
       
        #One electron energy
        E_1 = 2.0 * np.einsum('ij, ij->', self.H_core, P)
        E_1 += E_nuc
        
        E_old = 0
        F_list = []
        DIIS_error = []
        
        for i in range(1, self.maxiter+1):
            
            #term = 0
            #for k in range(self.I.shape[0]):
            #    for l in range(self.I.shape[0]):
            #        P_test = 0
            #        for a in range(2):
            #            P_test += C[k, a] * C[l, a]
            #        term += P_test * self.I[4, 3, k, l]

            J = np.einsum('ijkl, kl->ij', self.I, P)
            K = np.einsum('ikjl, kl->ij', self.I, P)

            F = self.H_core + 2.0 * J - K
            
            diis_e = np.einsum('ij,jk,kl->il', F, P, self.S) - np.einsum('ij,jk,kl->il', self.S, P, F)
            diis_e = X.T @ diis_e @ X
            F_list.append(F)
            DIIS_error.append(diis_e)
            dRMS = np.mean(diis_e**2)**0.5

            F_p = X.T @ F @ X
            eig, C_p = np.linalg.eigh(F_p)

            C = X @ C_p
            C_occ = C[:, :occ]

            E = np.einsum('ij, ij->', self.H_core + F, P) + E_nuc

            print(f"Iteration {i}: Energy = {E} Eh  dE = {E - E_old}  dRMS = {dRMS}")
            if (abs(E - E_old) < self.E_conv) and (dRMS < self.D_conv):
                break
            
            E_old = E

            if i >= 2:

                diis_count = len(F_list)
                if diis_count > 6:
                    del F_list[0]
                    del DIIS_error[0]
                    diis_count -= 1

                B = np.zeros((diis_count+1, diis_count+1))
                B[-1, :] = -1
                B[:, -1] = -1
                B[-1, -1] = 0
                for m, e1 in enumerate(DIIS_error):
                    for n, e2 in enumerate(DIIS_error):
                        if n > m: continue
                        val = np.einsum('ij, ij->', e1, e2)
                        B[m, n] = B[n, m] = val

                B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

                residual = np.zeros(diis_count + 1)
                residual[-1] = -1

                ci = np.linalg.solve(B, residual)

                F = np.zeros_like(F)
                for num, c in enumerate(ci[:-1]):
                    F += c * F_list[num]
            
            F_p = X.T @ F @ X
            eig, C_p = np.linalg.eigh(F_p)
            orbital_e = C_p.T @ F_p @ C_p
            C = X @ C_p
            C_occ = C[:, :occ]
            P = C_occ @ C_occ.T

        self.orbital_e = np.diag(orbital_e)
        self.C = C
        self.P_alpha = P
        self.C_occ = occ
        J = np.einsum('ijkl, kl->ij', self.I, P)
        K = np.einsum('ikjl, kl->ij', self.I, P)
        
        """
        eri_test = np.zeros_like(self.I)
        for a in range(self.I.shape[0]):
            for j in range(self.I.shape[0]):
                for k in range(self.I.shape[0]):
                    for l in range(self.I.shape[0]):
                        #P_2 = 0
                        #for a in range(C.shape[1]):
                        #    P_2 += C[j, a]
                        #P_3 = 0
                        #for a in range(C.shape[1]):
                        #    P_3 += C[k, a]
                        #P_4 = 0
                        #for a in range(C.shape[1]):
                        #    P_4 += C[l, a]
                        #eri_test[i, j, k, l] += P_1 * P_2 * P_3 * P_4 * self.I[i, j, k, l]
                        sum_val = 0.0
                        for i in range(self.I.shape[0]):
                            sum_val += C[i, a] * self.I[i, j, k, l]
                        eri_test[a, j, k, l] = sum_val
        """

        return E

    def calculate_gradient(self):
        """
        Calculate SCF gradient
        """
        grad_S = np.zeros((3, self.basis.pos.shape[0], self.basis.num_func, self.basis.num_func))
        grad_T = np.zeros((3, self.basis.pos.shape[0], self.basis.num_func, self.basis.num_func))
        grad_V = np.zeros((3, self.basis.pos.shape[0], self.basis.num_func, self.basis.num_func))
        grad_ERI = np.zeros((3, self.basis.pos.shape[0], self.basis.num_func, self.basis.num_func, self.basis.num_func, \
                self.basis.num_func))
        grad_ERI_test = np.load("grad_ERI.npy")
        grad_nuclear = np.zeros((3, self.basis.pos.shape[0]))
        for atom in range(self.basis.pos.shape[0]):
            for cart in range(3):
                S_dr = self.overlap.calculate_gradient(atom, cart)
                grad_S[cart, atom] = S_dr
            
            for cart in range(3):
                T_dr = self.kinetic.calculate_gradient(atom, cart)
                grad_T[cart, atom] = T_dr
        
            for cart in range(3):
                V_dr = self.potential.calculate_gradient(atom, cart)
                grad_V[cart, atom] = V_dr

            for cart in range(3):
                ERI_dr = self.eri.calculate_gradient(atom, cart)
                grad_ERI[cart, atom] = ERI_dr

            for cart in range(3):
                nuclear_dr = self.nuclear_repulsion.calculate_gradient(atom, cart)
                grad_nuclear[cart, atom] = nuclear_dr

        grad_Hcore = grad_T + grad_V
        grad_Hcore = 2.0 * np.einsum('ijuv, uv->ij', grad_Hcore, self.P_alpha)
        np.save("grad_ERI.npy", grad_ERI)
        
        grad_J = 2.0 * np.einsum('ijuvls, ls->ijuv', grad_ERI, self.P_alpha)
        grad_K = 2.0 * np.einsum('ijulvs, ls->ijuv', grad_ERI, self.P_alpha)

        grad_TEI = 2.0 * grad_J - grad_K
       
        C1_dr = np.load("C1_H4.npy")
        C1_dr[:, :, :, 1] *= -1
        C1_dr = C1_dr[0, 0]
        P_alpha_dr = np.einsum('ij,jk->ik', C1_dr, self.C[:, :self.C_occ].T)
        P_alpha_dr += np.einsum('ij,jk->ik', self.C[:, :self.C_occ], C1_dr.T)
        
        F = self.T + self.V + 2.0 * np.einsum('ijkl,kl->ij', self.I, self.P_alpha) - np.einsum('ikjl,kl->ij', self.I, self.P_alpha)

        F_dr = grad_T[0, 0] + grad_V[0, 0]

        F_dr += 2.0 * np.einsum('uvls, ls->uv', grad_ERI[0, 0], self.P_alpha)
        F_dr += 2.0 * np.einsum('ijkl,kl->ij', self.I, P_alpha_dr)

        F_dr -= np.einsum('ulvs, ls->uv', grad_ERI[0, 0], self.P_alpha)
        F_dr -= np.einsum('ikjl, kl->ij', self.I, P_alpha_dr)

        grad_TEI = 0.5 * np.einsum('ijuv, uv->ij', grad_TEI, self.P_alpha)

        W = np.einsum('i,ui,vi->uv', self.orbital_e[:self.C_occ], self.C[:, :self.C_occ], self.C[:, :self.C_occ])
        grad_S = -2.0 * np.einsum('abij,ij->ab', grad_S, W)
        
        grad = grad_Hcore + grad_TEI + grad_nuclear + grad_S
        grad = grad.transpose(1, 0)
        return grad
