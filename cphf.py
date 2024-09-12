import numpy as np
import scipy 

class CPHF:
    def __init__(self, overlap, kinetic, potential, eri, nuclear_repulsion, rhf, basis):
        self.overlap = overlap
        self.kinetic = kinetic
        self.potential = potential
        self.eri = eri
        self.rhf = rhf
        self.nuclear_repulsion = nuclear_repulsion
        self.basis = basis

    def calculate(self):
        
        occ = self.rhf.C_occ
        virtual = self.rhf.C.shape[0] - occ
        u = np.ones((virtual, occ))
        e = self.rhf.orbital_e
        eri = self.rhf.I
        
        grad_S = np.zeros((3, self.basis.pos.shape[0], self.basis.num_func, self.basis.num_func))
        grad_T = np.zeros((3, self.basis.pos.shape[0], self.basis.num_func, self.basis.num_func))
        grad_V = np.zeros((3, self.basis.pos.shape[0], self.basis.num_func, self.basis.num_func))
        grad_ERI = np.zeros((3, self.basis.pos.shape[0], self.basis.num_func, self.basis.num_func, self.basis.num_func, \
                self.basis.num_func))
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
       
        C = self.rhf.C
        grad_S = np.einsum('abij,jk->abik', grad_S, C)
        grad_S = np.einsum('ij, abjk->abik', C.T, grad_S)

        grad_H = grad_T + grad_V
        grad_H = np.einsum('abij, jk->abik', grad_H, C)
        grad_H = np.einsum('ij, abjk->abik', C.T, grad_H)

        eia = 1/(e[None, 0:occ] - e[occ:occ+virtual, None])
        
        C = self.rhf.C
        eri = self.rhf.I
        
        U = np.zeros((grad_S.shape[0], grad_S.shape[1], C.shape[0], C.shape[1]))
        U_occ = -1/2 * grad_S[:, :, :occ, :occ]
        U_diag = np.diagonal(U_occ, axis1=2, axis2=3).flatten()
        ind1 = np.arange(0, grad_S.shape[0])
        ind1 = np.repeat(ind1, int(U_diag.shape[0]/ind1.shape[0]))
        #3 dimensions
        ind2 = np.arange(0, 3)
        ind2 = np.repeat(ind2, (occ,))
        ind2 = np.tile(ind2, int(U_diag.shape[0]/ind2.shape[0]))
        ind3 = np.arange(occ)
        ind3 = np.tile(ind3, (int(U_diag.shape[0]/ind3.shape[0]),))

        U_occ = np.zeros_like(U_occ)
        U_occ[(ind1, ind2, ind3, ind3)] = U_diag

        G_2 = 2.0 * np.einsum('po, qj, rk, sk, abpqrs->aboj', C[:, occ:], C[:, :occ], C[:, :occ], C[:, :occ], grad_ERI)
        G_2 -= np.einsum('po, qk, rk, sj, abpqrs->aboj', C[:, occ:], C[:, :occ], C[:, :occ], C[:, :occ], grad_ERI)

        G_3 = 2.0 * np.einsum('abkl, po, qj, rl, sk, pqrs->aboj', grad_S[:, :, :occ, :occ], C[:, occ:], C[:, :occ], C[:, :occ], C[:, :occ], eri)
        G_3 -= np.einsum('abkl, po, qk, rl, sj, pqrs->aboj', grad_S[:, :, :occ, :occ], C[:, occ:], C[:, :occ], C[:, :occ], C[:, :occ], eri)

        def optimize_U(U):
            U = U.reshape(grad_S.shape[0], grad_S.shape[1], virtual, occ)

            G_1 = 4.0 * np.einsum('po, qj, rk, st, pqrs, abtk->aboj', C[:, occ:], C[:, :occ], C[:, :occ], C[:, occ:], eri, U)
            G_1 -= np.einsum('po, qk, rt, sj, pqrs, abtk->aboj', C[:, occ:], C[:, :occ], C[:, occ:], C[:, :occ], eri, U)
            G_1 -= np.einsum('po, qt, rk, sj, pqrs, abtk->aboj', C[:, occ:], C[:, occ:], C[:, :occ], C[:, :occ], eri, U)

            val = np.zeros_like(U)
            for o in range(occ, occ+virtual):
                for j in range(occ):
                    val[:, :, o-occ, j] = (grad_H[:, :, o, j] - e[j] * grad_S[:, :, o, j]) + G_1[:, :, o-occ, j] + G_2[:, :, o-occ, j] - G_3[:, :, o-occ, j]
            
                    val[:, :, o-occ, j] /= (e[j] - e[o])

            U = U.flatten()
            val = val.flatten()
            return ((U - val)**2).sum()

        def calculate_U_occ_ij(U_vir_occ):

            U_occ = np.zeros((grad_S.shape[0], grad_S.shape[1], occ, occ))

            G_1 = 4.0 * np.einsum('pi, qj, rk, st, pqrs, abtk->abij', C[:, :occ], C[:, :occ], C[:, :occ], C[:, occ:], eri, U_vir_occ)
            G_1 -= np.einsum('pi, qk, rt, sj, pqrs, abtk->abij', C[:, :occ], C[:, :occ], C[:, occ:], C[:, :occ], eri, U_vir_occ)
            G_1 -= np.einsum('pi, qt, rk, sj, pqrs, abtk->abij', C[:, :occ], C[:, occ:], C[:, :occ], C[:, :occ], eri, U_vir_occ)

            G_2 = 2.0 * np.einsum('pi, qj, rk, sk, abpqrs->abij', C[:, :occ], C[:, :occ], C[:, :occ], C[:, :occ], grad_ERI)
            G_2 -= np.einsum('pi, qk, rk, sj, abpqrs->abij', C[:, :occ], C[:, :occ], C[:, :occ], C[:, :occ], grad_ERI)

            G_3 = 2.0 * np.einsum('abkl, pi, qj, rl, sk, pqrs->abij', grad_S[:, :, :occ, :occ], C[:, :occ], C[:, :occ], C[:, :occ], C[:, :occ], eri)
            G_3 -= np.einsum('abkl, pi, qk, rl, sj, pqrs->abij', grad_S[:, :, :occ, :occ], C[:, :occ], C[:, :occ], C[:, :occ], C[:, :occ], eri)

            for i in range(occ):
                for j in range(i+1, occ):
                    print(i, j)
                    U_occ[:, :, i, j] = (grad_H[:, :, i, j] - e[j] * grad_S[:, :, i, j]) + G_1[:, :, i, j] + G_2[:, :, i, j] - G_3[:, :, i, j]

                    U_occ[:, :, i, j] /= (e[j] - e[i])
                   
                    U_occ[:, :, j, i] = -grad_S[:, :, j, i] - U_occ[:, :, i, j]
            
            return U_occ

        U_vir_occ = np.zeros((grad_S.shape[0], grad_S.shape[1], virtual, occ))
        
        U_vir_occ = U_vir_occ.flatten()
        
        result = scipy.optimize.minimize(optimize_U, U_vir_occ, method='BFGS', tol=1e-8)
        print(result.success)

        U_vir_occ = result.x

        U_vir_occ = U_vir_occ.reshape(grad_S.shape[0], grad_S.shape[1], virtual, occ)

        U_occ_ij = calculate_U_occ_ij(U_vir_occ)

        U[:, :, :occ, :occ] = U_occ

        U[:, :, :occ, :occ] += U_occ_ij
        
        U[:, :, occ:, :occ] = U_vir_occ

        C_1 = np.einsum('ij,abjk->abik', C, U)

        return C_1

