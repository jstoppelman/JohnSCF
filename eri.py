import numpy as np
import sys
from scipy.special import hyp1f1, erf
import itertools as it

class ERI:
    """
    electron electron repulsion tensor
    """
    def __init__(self, basis):
        
        self.basis = basis
        self.ERI = np.zeros((self.basis.num_func, self.basis.num_func, self.basis.num_func, self.basis.num_func))

    def index_added(self, idx_test, idx_set):
        """
        Tests the equivalent permutations for a given set of basis function indices and determines whether
        it has already been added to the basis function list

        Parameters
        -----------
        idx_test: list
            List of basis function indices which are set to be added 
        idx_set : list
            List of basis function indices that will be used for calculation of the ERI

        Returns
        ----------
        bool
            Determines whether idx_test has already been added to idx_set
        """

        idx_test2 = [idx_test[2], idx_test[3], idx_test[0], idx_test[1]]
        idx_test3 = [idx_test[1], idx_test[0], idx_test[3], idx_test[2]]
        idx_test4 = [idx_test[3], idx_test[2], idx_test[1], idx_test[0]]
        idx_test5 = [idx_test[1], idx_test[0], idx_test[2], idx_test[3]]
        idx_test6 = [idx_test[3], idx_test[2], idx_test[0], idx_test[1]]
        idx_test7 = [idx_test[0], idx_test[1], idx_test[3], idx_test[2]]
        idx_test8 = [idx_test[2], idx_test[3], idx_test[1], idx_test[0]]
       
        if (idx_test not in idx_set) and (idx_test2 not in idx_set) and (idx_test3 not in idx_set) and \
                (idx_test4 not in idx_set) and (idx_test5 not in idx_set) and (idx_test6 not in idx_set) and \
                (idx_test7 not in idx_set) and (idx_test8 not in idx_set):

            return False
        else:
            return True

    def _make_two_shell_pair(self):
        """
        Generate all possible pairs of basis functions
        """

        pairs = []
        pair_added = []
        for i in range(self.basis.wfn.basisset().nshell()):
            shell_i = self.basis.wfn.basisset().shell(i)
            index_mu = shell_i.function_index
            for j in range(self.basis.wfn.basisset().nshell()):
                shell_j = self.basis.wfn.basisset().shell(j)
                index_nu = shell_j.function_index
                for k in range(self.basis.wfn.basisset().nshell()):
                    shell_k = self.basis.wfn.basisset().shell(k)
                    index_lambda = shell_k.function_index
                    for l in range(self.basis.wfn.basisset().nshell()):
                        shell_l = self.basis.wfn.basisset().shell(l)
                        index_sigma = shell_l.function_index
                        function_indices = [index_mu, index_nu, index_lambda, index_sigma]
                        if not self.index_added(function_indices, pair_added):
                            pairs.append([shell_i, shell_j, shell_k, shell_l])
                            pair_added.append(function_indices)
        
        return pairs

    def calculate_R(self, RPC, p, n):
        """
        Calculates F value needed for calculating three center integrals

        Parameters
        -----------
        RPC: np.array
            Vector between P (weighted center of Gaussians) and nuclei (C)
        p: np.array
            Sum of integral values
        n: int
            Order for computing hypergeometric function

        Returns
        -----------
        F: np.array
            hypergeometric function values
        """

        x = (RPC**2).sum(axis=-1) * p
        F = hyp1f1(n+0.5, n+1.5, -x) / (2 * n + 1)
        return F

    def recursive_theta_i(self, 
            xyz_mu,
            xyz_nu,
            xyz_lambda,
            xyz_sigma,
            RPQ_coords,
            PA_coords,
            exp_list,
            am_x,
            am_y,
            am_z,
            N
            ):
        """
        First recursion: recursion over "g" which is the sum of angular momentum for each basis function

        Parameters
        -----------
        xyz_mu: np.ndarray
            First basis function center
        xyz_nu: np.ndarray
            Second basis function center
        xyz_lambda: np.ndarray
            Third basis function center
        xyz_sigma: np.ndarray
            Fourth basis function center
        RPQ_coords: np.ndarray
            Distance between centers of both pairs of basis functions
        PA_coords: np.ndarray
            Distance between basis function center P and first basis function center
        exp_list: np.ndarray
            Basis function exponents
        am_x, am_y, am_z: int
            Angular momentum in each direction
        N: int
            Order of hypergeometric function

        Returns
        -----------
        val: np.ndarray
            First recursion g value
        """
      
        val = np.zeros((RPQ_coords.shape[0]))
        p = exp_list[:, 0:2].sum(axis=1)
        q = exp_list[:, 2:].sum(axis=1)
        alpha = (p*q)/(p + q)

        if am_x == am_y == am_z == 0:
            RAB = ((xyz_mu - xyz_nu)**2).sum(axis=0)
            mu_ab = (np.prod(exp_list[:, 0:2], axis=1))/(exp_list[:, 0:2].sum(axis=1))
            K_AB = np.exp(-mu_ab * RAB)
            RCD = ((xyz_lambda - xyz_sigma)**2).sum(axis=0)
            mu_cd = (np.prod(exp_list[:, 2:], axis=1))/(exp_list[:, 2:].sum(axis=1))
            K_CD = np.exp(-mu_cd * RCD)
            
            R = self.calculate_R(RPQ_coords, alpha, N)
            
            val += 2.0 * np.pi**(5/2) / (p * q * np.sqrt(p + q)) * K_AB * K_CD * R

        elif am_x > 0:
            val += PA_coords[:, 0] * self.recursive_theta_i(xyz_mu, 
                                                            xyz_nu, 
                                                            xyz_lambda, 
                                                            xyz_sigma, 
                                                            RPQ_coords, 
                                                            PA_coords,       
                                                            exp_list, 
                                                            am_x-1, 
                                                            am_y, 
                                                            am_z, 
                                                            N) - \
                   alpha/p * RPQ_coords[:, 0] * self.recursive_theta_i(xyz_mu, 
                                                                       xyz_nu, 
                                                                       xyz_lambda, 
                                                                       xyz_sigma, 
                                                                       RPQ_coords, 
                                                                       PA_coords,
                                                                       exp_list, 
                                                                       am_x-1, 
                                                                       am_y, 
                                                                       am_z, 
                                                                       N+1) + \
                   (am_x-1)/p * 0.5 * (self.recursive_theta_i(xyz_mu, 
                                                              xyz_nu, 
                                                              xyz_lambda, 
                                                              xyz_sigma, 
                                                              RPQ_coords, 
                                                              PA_coords,
                                                              exp_list, 
                                                              am_x-2, 
                                                              am_y, 
                                                              am_z, 
                                                              N) - \
                   alpha/p * self.recursive_theta_i(xyz_mu, 
                                                    xyz_nu, 
                                                    xyz_lambda, 
                                                    xyz_sigma, 
                                                    RPQ_coords, 
                                                    PA_coords, 
                                                    exp_list, 
                                                    am_x-2, 
                                                    am_y, 
                                                    am_z, 
                                                    N+1))
        elif am_y > 0:
            val += PA_coords[:, 1] * self.recursive_theta_i(xyz_mu, 
                                                            xyz_nu, 
                                                            xyz_lambda, 
                                                            xyz_sigma, 
                                                            RPQ_coords, 
                                                            PA_coords,
                                                            exp_list, 
                                                            am_x, 
                                                            am_y-1, 
                                                            am_z, 
                                                            N) - \
                   alpha/p * RPQ_coords[:, 1] * self.recursive_theta_i(xyz_mu, 
                                                                       xyz_nu, 
                                                                       xyz_lambda, 
                                                                       xyz_sigma, 
                                                                       RPQ_coords, 
                                                                       PA_coords,
                                                                       exp_list, 
                                                                       am_x, 
                                                                       am_y-1, 
                                                                       am_z, 
                                                                       N+1) + \
                   (am_y-1)/p * 0.5 * (self.recursive_theta_i(xyz_mu, 
                                                              xyz_nu, 
                                                              xyz_lambda, 
                                                              xyz_sigma, 
                                                              RPQ_coords, 
                                                              PA_coords,
                                                              exp_list, 
                                                              am_x, 
                                                              am_y-2, 
                                                              am_z, 
                                                              N) - \
                   alpha/p * self.recursive_theta_i(xyz_mu, 
                                                    xyz_nu, 
                                                    xyz_lambda, 
                                                    xyz_sigma,
                                                    RPQ_coords, 
                                                    PA_coords, 
                                                    exp_list, 
                                                    am_x, 
                                                    am_y-2, 
                                                    am_z, 
                                                    N+1))
        elif am_z > 0:
            val += PA_coords[:, 2] * self.recursive_theta_i(xyz_mu, 
                                                            xyz_nu, 
                                                            xyz_lambda, 
                                                            xyz_sigma, 
                                                            RPQ_coords, 
                                                            PA_coords,
                                                            exp_list, 
                                                            am_x, 
                                                            am_y, 
                                                            am_z-1, 
                                                            N) - \
                   alpha/p * RPQ_coords[:, 2] * self.recursive_theta_i(xyz_mu, 
                                                                       xyz_nu, 
                                                                       xyz_lambda, 
                                                                       xyz_sigma, 
                                                                       RPQ_coords, 
                                                                       PA_coords,
                                                                       exp_list, 
                                                                       am_x, 
                                                                       am_y, 
                                                                       am_z-1, 
                                                                       N+1) + \
                    (am_z-1)/p * 0.5 * (self.recursive_theta_i(xyz_mu, 
                                                               xyz_nu, 
                                                               xyz_lambda, 
                                                               xyz_sigma, 
                                                               RPQ_coords, 
                                                               PA_coords,
                                                               exp_list, 
                                                               am_x, 
                                                               am_y, 
                                                               am_z-2, 
                                                               N) - \
                    alpha/p * self.recursive_theta_i(xyz_mu, 
                                                     xyz_nu, 
                                                     xyz_lambda, 
                                                     xyz_sigma,
                                                     RPQ_coords, 
                                                     PA_coords, 
                                                     exp_list, 
                                                     am_x, 
                                                     am_y, 
                                                     am_z-2, 
                                                     N+1))

        return val

    def recursive_electron_transfer(self,
            am_i_integrals,
            xyz_mu,
            xyz_nu,
            xyz_lambda,
            xyz_sigma,
            RPQ_coords,
            exp_list,
            am_i_x,
            am_i_y,
            am_i_z,
            am_k_x,
            am_k_y,
            am_k_z,
            N
            ):
        """
        Second recursion: transfer from i basis function to k basis function

        Parameters
        -----------
        am_i_integrals: dict
            Dictionary containing computed g AM values from the previous recursion step
        xyz_mu: np.ndarray
            First basis function center
        xyz_nu: np.ndarray
            Second basis function center
        xyz_lambda: np.ndarray
            Third basis function center
        xyz_sigma: np.ndarray
            Fourth basis function center
        RPQ_coords: np.ndarray
            Distance between centers of both pairs of basis functions
        exp_list: np.ndarray
            Basis function centers
        am_i_x, am_i_y, am_i_z, am_k_x, am_k_x, am_k_y, am_k_z: int
            Combined angular momentum of both first and second pair of basis functions
        N: int
            Order of hypergeometric function
    
        Returns
        -----------
        val: np.npdarray
            Value of second recursion step
        """
        b = exp_list[:, 1]
        d = exp_list[:, 3]
        p = exp_list[:, 0:2].sum(axis=1)
        q = exp_list[:, 2:].sum(axis=1)
        val = np.zeros((RPQ_coords.shape[0]))

        if am_k_x == am_k_y == am_k_z == 0:
            if tuple((am_i_x, am_i_y, am_i_z)) in am_i_integrals.keys():
                val += am_i_integrals[(am_i_x, am_i_y, am_i_z)]

        elif am_k_x > 0:
            XAB = xyz_mu[0] - xyz_nu[0]
            XCD = xyz_lambda[0] - xyz_sigma[0]
            val += -(b * XAB + d * XCD)/q * self.recursive_electron_transfer(am_i_integrals, 
                                                                             xyz_mu, 
                                                                             xyz_nu, 
                                                                             xyz_lambda, 
                                                                             xyz_sigma,
                                                                             RPQ_coords,
                                                                             exp_list, 
                                                                             am_i_x, 
                                                                             am_i_y, 
                                                                             am_i_z, 
                                                                             am_k_x-1, 
                                                                             am_k_y, 
                                                                             am_k_z, 
                                                                             N) + \
                   am_i_x/q * 0.5 * self.recursive_electron_transfer(am_i_integrals, 
                                                                     xyz_mu, 
                                                                     xyz_nu, 
                                                                     xyz_lambda, 
                                                                     xyz_sigma, 
                                                                     RPQ_coords,
                                                                     exp_list, 
                                                                     am_i_x-1, 
                                                                     am_i_y, 
                                                                     am_i_z, 
                                                                     am_k_x-1, 
                                                                     am_k_y, 
                                                                     am_k_z, 
                                                                     N) + \
                   (am_k_x-1)/q * 0.5 * self.recursive_electron_transfer(am_i_integrals, 
                                                                         xyz_mu, 
                                                                         xyz_nu, 
                                                                         xyz_lambda, 
                                                                         xyz_sigma, 
                                                                         RPQ_coords,
                                                                         exp_list, 
                                                                         am_i_x, 
                                                                         am_i_y, 
                                                                         am_i_z, 
                                                                         am_k_x-2, 
                                                                         am_k_y, 
                                                                         am_k_z, 
                                                                         N) - \
                   p/q * self.recursive_electron_transfer(am_i_integrals, 
                                                          xyz_mu, 
                                                          xyz_nu, 
                                                          xyz_lambda, 
                                                          xyz_sigma, 
                                                          RPQ_coords,
                                                          exp_list, 
                                                          am_i_x+1, 
                                                          am_i_y, 
                                                          am_i_z, 
                                                          am_k_x-1, 
                                                          am_k_y, 
                                                          am_k_z, 
                                                          N)
        
        elif am_k_y > 0:
            YAB = xyz_mu[1] - xyz_nu[1]
            YCD = xyz_lambda[1] - xyz_sigma[1]
            val += -(b * YAB + d * YCD)/q * self.recursive_electron_transfer(am_i_integrals, 
                                                                             xyz_mu, 
                                                                             xyz_nu, 
                                                                             xyz_lambda, 
                                                                             xyz_sigma, 
                                                                             RPQ_coords,
                                                                             exp_list, 
                                                                             am_i_x, 
                                                                             am_i_y, 
                                                                             am_i_z, 
                                                                             am_k_x, 
                                                                             am_k_y-1, 
                                                                             am_k_z, 
                                                                             N) + \
                   am_i_y/q * 0.5 * self.recursive_electron_transfer(am_i_integrals, 
                                                                     xyz_mu, 
                                                                     xyz_nu, 
                                                                     xyz_lambda, 
                                                                     xyz_sigma, 
                                                                     RPQ_coords,
                                                                     exp_list, 
                                                                     am_i_x, 
                                                                     am_i_y-1, 
                                                                     am_i_z, 
                                                                     am_k_x, 
                                                                     am_k_y-1, 
                                                                     am_k_z, 
                                                                     N) + \
                   (am_k_y-1)/q * 0.5 * self.recursive_electron_transfer(am_i_integrals, 
                                                                         xyz_mu, 
                                                                         xyz_nu, 
                                                                         xyz_lambda, 
                                                                         xyz_sigma, 
                                                                         RPQ_coords,
                                                                         exp_list, 
                                                                         am_i_x, 
                                                                         am_i_y, 
                                                                         am_i_z, 
                                                                         am_k_x, 
                                                                         am_k_y-2, 
                                                                         am_k_z, 
                                                                         N) - \
                   p/q * self.recursive_electron_transfer(am_i_integrals, 
                                                          xyz_mu, 
                                                          xyz_nu, 
                                                          xyz_lambda, 
                                                          xyz_sigma, 
                                                          RPQ_coords,
                                                          exp_list, 
                                                          am_i_x, 
                                                          am_i_y+1, 
                                                          am_i_z, 
                                                          am_k_x, 
                                                          am_k_y-1, 
                                                          am_k_z, 
                                                          N)

        elif am_k_z > 0:
            ZAB = xyz_mu[2] - xyz_nu[2]
            ZCD = xyz_lambda[2] - xyz_sigma[2]
            val += -(b * ZAB + d * ZCD)/q * self.recursive_electron_transfer(am_i_integrals, 
                                                                             xyz_mu, 
                                                                             xyz_nu, 
                                                                             xyz_lambda, 
                                                                             xyz_sigma, 
                                                                             RPQ_coords,
                                                                             exp_list, 
                                                                             am_i_x, 
                                                                             am_i_y, 
                                                                             am_i_z, 
                                                                             am_k_x, 
                                                                             am_k_y, 
                                                                             am_k_z-1, 
                                                                             N) + \
                   am_i_z/q * 0.5 * self.recursive_electron_transfer(am_i_integrals, 
                                                                     xyz_mu, 
                                                                     xyz_nu, 
                                                                     xyz_lambda, 
                                                                     xyz_sigma, 
                                                                     RPQ_coords,
                                                                     exp_list, 
                                                                     am_i_x, 
                                                                     am_i_y, 
                                                                     am_i_z-1, 
                                                                     am_k_x, 
                                                                     am_k_y, 
                                                                     am_k_z-1, 
                                                                     N) + \
                   (am_k_z-1)/q * 0.5 * self.recursive_electron_transfer(am_i_integrals, 
                                                                         xyz_mu, 
                                                                         xyz_nu, 
                                                                         xyz_lambda, 
                                                                         xyz_sigma, 
                                                                         RPQ_coords,
                                                                         exp_list, 
                                                                         am_i_x, 
                                                                         am_i_y, 
                                                                         am_i_z, 
                                                                         am_k_x, 
                                                                         am_k_y, 
                                                                         am_k_z-2, 
                                                                         N) - \
                   p/q * self.recursive_electron_transfer(am_i_integrals, 
                                                          xyz_mu, 
                                                          xyz_nu, 
                                                          xyz_lambda, 
                                                          xyz_sigma, 
                                                          RPQ_coords,
                                                          exp_list, 
                                                          am_i_x, 
                                                          am_i_y, 
                                                          am_i_z+1, 
                                                          am_k_x, 
                                                          am_k_y, 
                                                          am_k_z-1, 
                                                          N)

        return val

    def recursive_horizontal_transfer_j(self,
            am_integrals,
            xyz_a,
            xyz_b,
            am_i_x,
            am_i_y,
            am_i_z,
            am_j_x,
            am_j_y,
            am_j_z,
            am_f_x,
            am_f_y,
            am_f_z,
            ):
        """
        Third recursion: transfer from i basis function to j basis function

        Parameters
        -----------
        am_integrals: dict
            Dictionary containing computed ik AM values from the previous recursion step
        xyz_a: np.ndarray
            First basis function center (of i)
        xyz_b: np.ndarray
            Second basis function center (j)
        am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z, am_f_x, am_f_y, am_f_z: int
            Angular momentum of basis function i, j and combined angular momentum of k and l (f)

        Returns
        -----------
        val: np.npdarray
            Value of third recursion step
        """

        val = 0
        xyz = xyz_a - xyz_b
        if am_j_x == am_j_y == am_j_z == 0:
            key = tuple((am_i_x, am_i_y, am_i_z, am_f_x, am_f_y, am_f_z))
            val += am_integrals[key]

        elif am_j_x > 0:
            val += self.recursive_horizontal_transfer_j(am_integrals,
                                                        xyz_a,
                                                        xyz_b,
                                                        am_i_x+1,
                                                        am_i_y,
                                                        am_i_z,
                                                        am_j_x-1,
                                                        am_j_y,
                                                        am_j_z,
                                                        am_f_x,
                                                        am_f_y,
                                                        am_f_z) + \
                   xyz[0] * self.recursive_horizontal_transfer_j(am_integrals,
                                                                 xyz_a,
                                                                 xyz_b,
                                                                 am_i_x,
                                                                 am_i_y,
                                                                 am_i_z,
                                                                 am_j_x-1,
                                                                 am_j_y,
                                                                 am_j_z,
                                                                 am_f_x,
                                                                 am_f_y,
                                                                 am_f_z)

        elif am_j_y > 0:
            val += self.recursive_horizontal_transfer_j(am_integrals,
                                                        xyz_a,
                                                        xyz_b,
                                                        am_i_x,
                                                        am_i_y+1,
                                                        am_i_z,
                                                        am_j_x,
                                                        am_j_y-1,
                                                        am_j_z,
                                                        am_f_x,
                                                        am_f_y,
                                                        am_f_z) + \
                   xyz[1] * self.recursive_horizontal_transfer_j(am_integrals,
                                                                 xyz_a,
                                                                 xyz_b,
                                                                 am_i_x,
                                                                 am_i_y,
                                                                 am_i_z,
                                                                 am_j_x,
                                                                 am_j_y-1,
                                                                 am_j_z,
                                                                 am_f_x,
                                                                 am_f_y,
                                                                 am_f_z)

        elif am_j_z > 0:
            val += self.recursive_horizontal_transfer_j(am_integrals,
                                                        xyz_a,
                                                        xyz_b,
                                                        am_i_x,
                                                        am_i_y,
                                                        am_i_z+1,
                                                        am_j_x,
                                                        am_j_y,
                                                        am_j_z-1,
                                                        am_f_x,
                                                        am_f_y,
                                                        am_f_z) + \
                   xyz[2] * self.recursive_horizontal_transfer_j(am_integrals,
                                                                 xyz_a,
                                                                 xyz_b,
                                                                 am_i_x,
                                                                 am_i_y,
                                                                 am_i_z,
                                                                 am_j_x,
                                                                 am_j_y,
                                                                 am_j_z-1,
                                                                 am_f_x,
                                                                 am_f_y,
                                                                 am_f_z)

        return val

    def recursive_horizontal_transfer_l(self,
            am_integrals,
            xyz_a,
            xyz_b,
            am_i_x,
            am_i_y,
            am_i_z,
            am_j_x,
            am_j_y,
            am_j_z,
            am_k_x,
            am_k_y,
            am_k_z,
            am_l_x,
            am_l_y,
            am_l_z
            ):
        """
        Fourth recursion: transfer from f basis function to l basis function

        Parameters
        -----------
        am_integrals: dict
            Dictionary containing computed ijf AM values from the previous recursion step
        xyz_a: np.ndarray
            First basis function center (of i)
        xyz_b: np.ndarray
            Second basis function center (j)
        am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z, am_f_x, am_f_y, am_f_z: int
            Angular momentum of basis function i, j and combined angular momentum of k and l (f)

        Returns
        -----------
        val: np.npdarray
            Value of final recursion step
        """

        val = 0
        xyz = xyz_a - xyz_b
        if am_l_x == am_l_y == am_l_z == 0:
            key = tuple((am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z, am_k_x, am_k_y, am_k_z))
            val += am_integrals[key]

        elif am_l_x > 0:
            val += self.recursive_horizontal_transfer_l(am_integrals,
                                                        xyz_a,
                                                        xyz_b,
                                                        am_i_x,
                                                        am_i_y,
                                                        am_i_z,
                                                        am_j_x,
                                                        am_j_y,
                                                        am_j_z,
                                                        am_k_x+1,
                                                        am_k_y,
                                                        am_k_z,
                                                        am_l_x-1,
                                                        am_l_y,
                                                        am_l_z) + \
                   xyz[0] * self.recursive_horizontal_transfer_l(am_integrals,
                                                                 xyz_a,
                                                                 xyz_b,
                                                                 am_i_x,
                                                                 am_i_y,
                                                                 am_i_z,
                                                                 am_j_x,
                                                                 am_j_y,
                                                                 am_j_z,
                                                                 am_k_x,
                                                                 am_k_y,
                                                                 am_k_z,
                                                                 am_l_x-1,
                                                                 am_l_y,
                                                                 am_l_z)

        elif am_l_y > 0:
            val += self.recursive_horizontal_transfer_l(am_integrals,
                                                        xyz_a,
                                                        xyz_b,
                                                        am_i_x,
                                                        am_i_y,
                                                        am_i_z,
                                                        am_j_x,
                                                        am_j_y,
                                                        am_j_z,
                                                        am_k_x,
                                                        am_k_y+1,
                                                        am_k_z,
                                                        am_l_x,
                                                        am_l_y-1,
                                                        am_l_z) + \
                   xyz[1] * self.recursive_horizontal_transfer_l(am_integrals,
                                                                 xyz_a,
                                                                 xyz_b,
                                                                 am_i_x,
                                                                 am_i_y,
                                                                 am_i_z,
                                                                 am_j_x,
                                                                 am_j_y,
                                                                 am_j_z,
                                                                 am_k_x,
                                                                 am_k_y,
                                                                 am_k_z,
                                                                 am_l_x,
                                                                 am_l_y-1,
                                                                 am_l_z)

        elif am_l_z > 0:
            val += self.recursive_horizontal_transfer_l(am_integrals,
                                                        xyz_a,
                                                        xyz_b,
                                                        am_i_x,
                                                        am_i_y,
                                                        am_i_z,
                                                        am_j_x,
                                                        am_j_y,
                                                        am_j_z,
                                                        am_k_x,
                                                        am_k_y,
                                                        am_k_z+1,
                                                        am_l_x,
                                                        am_l_y,
                                                        am_l_z-1) + \
                   xyz[2] * self.recursive_horizontal_transfer_l(am_integrals,
                                                                 xyz_a,
                                                                 xyz_b,
                                                                 am_i_x,
                                                                 am_i_y,
                                                                 am_i_z,
                                                                 am_j_x,
                                                                 am_j_y,
                                                                 am_j_z,
                                                                 am_k_x,
                                                                 am_k_y,
                                                                 am_k_z,
                                                                 am_l_x,
                                                                 am_l_y,
                                                                 am_l_z-1)
        return val

    def calculate_P_coords(self, exp_mu, exp_nu, xyz_mu, xyz_nu):
        """
        Return the P coordinate for computing overlap integrals

        Paramters
        ----------
        xyz_mu: np.array size 3
            coordinates of first center
        xyz_nu: np.array size 3
            coordinates of second center

        Returns
        ----------
        P_coord: list
            List of PXA, PXB, PYA etc.
        PC: np.ndarray
            Distance from coordinate P (weighted average of Gaussians) and all nuclei
        """
        P_xyz = (exp_mu * xyz_mu + exp_nu * xyz_nu) * 1/(exp_mu + exp_nu)
        PA = P_xyz - xyz_mu
        return P_xyz, PA

    def calculate_primitives(self, 
            func_1, 
            func_2,
            func_3,
            func_4,
            xyz_mu, 
            xyz_nu,
            xyz_lambda,
            xyz_sigma):
        """
        Calculate overlap integral between two basis functions

        Parameters
        -----------
        func_1: class
            Psi4 shell basis function
        func_2: class
            Psi4 shell basis function
        xyz_mu: np.array
            Containing positions of atom corresponding to first basis function
        xyz_nu: np.array
            Containing positions of atom corresponding to second basis function

        Returns
        -----------
        s00_ints: list
            Individual Gaussian primitive integrals
        coefs: list
            Basis function coefficients
        t00_ints: list
            Kinetic energy primitive integrals
        P_coords: list
            Needed for calculating higher momentum Gaussian integrals
        exp_list: list
            List of exponents for each basis function pair
        exp_sum_inv_list: list
            Sum of exponents for each integral
        """
        RPQ_coords = []
        PA_coords = []
        exp_list = []
        coefs = []
        for p1 in range(func_1.nprimitive):
            for p2 in range(func_2.nprimitive):
                for p3 in range(func_3.nprimitive):
                    for p4 in range(func_4.nprimitive):
                        coef_mu = func_1.coef(p1)
                        coef_nu = func_2.coef(p2)
                        coef_lambda = func_3.coef(p3)
                        coef_sigma = func_4.coef(p4)
                        coefs.append(coef_mu*coef_nu*coef_lambda*coef_sigma)
                        exp_mu = func_1.exp(p1)
                        exp_nu = func_2.exp(p2)
                        exp_lambda = func_3.exp(p3)
                        exp_sigma = func_4.exp(p4)
                        exp_list.append([exp_mu, exp_nu, exp_lambda, exp_sigma])

                        P_xyz, PA = self.calculate_P_coords(exp_mu, exp_nu, xyz_mu, xyz_nu)
                        Q_xyz, QA = self.calculate_P_coords(exp_lambda, exp_sigma, xyz_lambda, xyz_sigma)
                        RPQ_coords.append(P_xyz - Q_xyz)
                        PA_coords.append(PA)

        return coefs, exp_list, RPQ_coords, PA_coords

    def calculate_integral(self, 
            func_1, 
            func_2,
            func_3,
            func_4,
            xyz_mu, 
            xyz_nu,
            xyz_lambda,
            xyz_sigma):
        """
        Calculate ERI component

        Parameters
        -----------
        func_1, func_2, func_3, func_4: shell objects
            Shell objects from Psi4
        xyz: np.ndarray
            Center of basis functions

        Returns
        -----------
        integrals_total: np.ndarray
            ERI integral contribution from this set of basis functions
        """

        #Form the required integrals in each dimension (am_ints is the final variable)
        am_1 = func_1.am
        am_2 = func_2.am
        am_3 = func_3.am
        am_4 = func_4.am

        am_pair = np.asarray([am_1, am_2]).astype(int)
        am_pair2 = np.asarray([am_3, am_4]).astype(int)

        max_am = np.max(am_pair)
        am_values = np.arange(max_am+1)
        am_grid = np.asarray(np.meshgrid(am_values, am_values, am_values)).T.reshape(-1, 3)
        am_ints1 = am_grid[np.sum(am_grid, axis=1) == am_pair[0]]
        am_ints2 = am_grid[np.sum(am_grid, axis=1) == am_pair[1]]
        am_1 = np.tile(am_ints1, (am_ints2.shape[0], 1))
        am_2 = np.repeat(am_ints2, am_ints1.shape[0], axis=0)
        am_ints_uv = np.hstack((am_1, am_2))
        am_ints_uv = am_ints_uv.reshape(am_ints_uv.shape[0], 2, -1)

        am_ints_uv = self.basis.reorder_int(am_ints_uv, am_pair)

        max_am = np.max(am_pair2)
        am_values = np.arange(max_am+1)
        am_grid = np.asarray(np.meshgrid(am_values, am_values, am_values)).T.reshape(-1, 3)
        am_ints1 = am_grid[np.sum(am_grid, axis=1) == am_pair2[0]]
        am_ints2 = am_grid[np.sum(am_grid, axis=1) == am_pair2[1]]
        am_1 = np.tile(am_ints1, (am_ints2.shape[0], 1))
        am_2 = np.repeat(am_ints2, am_ints1.shape[0], axis=0)
        am_ints_ls = np.hstack((am_1, am_2))
        am_ints_ls = am_ints_ls.reshape(am_ints_ls.shape[0], 2, -1)

        am_ints_ls = self.basis.reorder_int(am_ints_ls, am_pair2)

        am_ints = np.asarray(list(it.product(am_ints_uv, am_ints_ls))).reshape(-1, 4, 3)

        coefs, exp_list, RPQ_coords, PA_coords = self.calculate_primitives(func_1,
                func_2, 
                func_3,
                func_4, 
                xyz_mu,
                xyz_nu,
                xyz_lambda,
                xyz_sigma
                )

        coefs = np.asarray(coefs)
        exp_list = np.asarray(exp_list)
        RPQ_coords = np.asarray(RPQ_coords)
        PA_coords = np.asarray(PA_coords)

        integral_total = np.zeros((am_ints.shape[0]))

        for i in range(am_ints.shape[0]):
            #Will always need (ss|ss) integral
            
            am_int = am_ints[i]
            am_g_x = am_int[:, 0].sum()
            am_g_y = am_int[:, 1].sum()
            am_g_z = am_int[:, 2].sum()
            am_g_integrals = {}
            
            am_set = list(it.product(list(range(am_g_x+1)), list(range(am_g_y+1)), list(range(am_g_z+1))))
            for integral in am_set:
                theta = self.recursive_theta_i(
                        xyz_mu,
                        xyz_nu,
                        xyz_lambda,
                        xyz_sigma,
                        RPQ_coords,
                        PA_coords,
                        exp_list,
                        integral[0],
                        integral[1],
                        integral[2],
                        0.0
                        )
                
                am_g_integrals[integral] = theta

            am_ef_integrals = {}

            am_e_x = am_int[0:2, 0].sum()
            am_e_y = am_int[0:2, 1].sum()
            am_e_z = am_int[0:2, 2].sum()
            am_f_x = am_int[2:, 0].sum()
            am_f_y = am_int[2:, 1].sum()
            am_f_z = am_int[2:, 2].sum()
            if np.asarray([am_f_x, am_f_y, am_f_z]).any():
                am_set_f = list(it.product(list(range(am_f_x+1)), list(range(am_f_y+1)), list(range(am_f_z+1))))
                am_set_e = list(it.product(list(range(am_e_x+1)), list(range(am_e_y+1)), list(range(am_e_z+1))))
                am_set_ef = list(it.product(am_set_e, am_set_f))
                
                for integral in am_set_ef:
                    theta = self.recursive_electron_transfer(
                        am_g_integrals,
                        xyz_mu,
                        xyz_nu,
                        xyz_lambda,
                        xyz_sigma,
                        RPQ_coords,
                        exp_list,
                        integral[0][0],
                        integral[0][1],
                        integral[0][2],
                        integral[1][0],
                        integral[1][1],
                        integral[1][2],
                        0.0
                        )

                    theta *= coefs
                    key = tuple((integral[0][0], integral[0][1], integral[0][2], integral[1][0], integral[1][1], integral[1][2]))
                    am_ef_integrals[key] = theta.sum()

            else:

                am_int_tmp = {}
                for key, val in am_g_integrals.items():
                    new_key = tuple((key[0], key[1], key[2], 0, 0, 0))
                    new_val = val * coefs
                    new_val = new_val.sum()
                    am_int_tmp[new_key] = new_val
                
                am_ef_integrals = am_int_tmp
         
            am_i_x, am_i_y, am_i_z = am_int[0, :]
            am_j_x, am_j_y, am_j_z = am_int[1, :]
            am_k_x, am_k_y, am_k_z = am_int[2, :]
            am_l_x, am_l_y, am_l_z = am_int[3, :]
            if np.asarray([am_j_x, am_j_y, am_j_z]).any():
                am_ijf_integrals = {}
                am_ij_x = am_i_x + am_j_x
                am_ij_y = am_i_y + am_j_y
                am_ij_z = am_i_z + am_j_z
                for key in am_ef_integrals.keys():
                    if key[0] == am_ij_x or key[1] == am_ij_y or key[2] == am_ij_z:
                        theta = self.recursive_horizontal_transfer_j(am_ef_integrals,
                                                                     xyz_mu, 
                                                                     xyz_nu,
                                                                     am_i_x,
                                                                     am_i_y,
                                                                     am_i_z,
                                                                     am_j_x,
                                                                     am_j_y,
                                                                     am_j_z,
                                                                     key[3],
                                                                     key[4],
                                                                     key[5])

                        key = tuple((am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z, key[3], key[4], key[5]))
                        am_ijf_integrals[key] = theta

            else:
                am_int_tmp = {}
                for key, val in am_ef_integrals.items():
                    new_key = tuple((key[0], key[1], key[2], 0, 0, 0, key[3], key[4], key[5]))
                    am_int_tmp[new_key] = val
                am_ijf_integrals = am_int_tmp

            am_ijkl_integrals = {}
            if np.asarray([am_l_x, am_l_y, am_l_z]).any():
                theta = self.recursive_horizontal_transfer_l(am_ijf_integrals,
                                                             xyz_lambda,
                                                             xyz_sigma,
                                                             am_i_x,
                                                             am_i_y,
                                                             am_i_z,
                                                             am_j_x,
                                                             am_j_y,
                                                             am_j_z,
                                                             am_k_x,
                                                             am_k_y,
                                                             am_k_z,
                                                             am_l_x,
                                                             am_l_y,
                                                             am_l_z
                                                             )
                key = tuple((am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z, 
                             am_k_x, am_k_y, am_k_z, am_l_x, am_l_y, am_l_z))
                am_ijkl_integrals[key] = theta
            else:
                am_int_tmp = {}
                for key, val in am_ijf_integrals.items():
                    new_key = tuple((key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7], key[8], 0, 0, 0))
                    am_int_tmp[new_key] = val
                am_ijkl_integrals = am_int_tmp
            
            key = tuple((am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z,
                         am_k_x, am_k_y, am_k_z, am_l_x, am_l_y, am_l_z))
            
            integral_total[i] = am_ijkl_integrals[key]

        return integral_total

    def calculate_integral_derivative(self,
            func_1,
            func_2,
            func_3,
            func_4,
            xyz_mu,
            xyz_nu,
            xyz_lambda,
            xyz_sigma,
            atom,
            component):
        """
        Calculate ERI gradient component

        Parameters
        -----------
        func_1, func_2, func_3, func_4: shell objects
            Shell objects from Psi4
        xyz: np.ndarray
            Center of basis functions
        atom: int
            Atom index
        component: int
            Cartesian component index

        Returns
        -----------
        integrals_total: np.ndarray
            ERI integral contribution from this set of basis functions
        """


        index_mu = func_1.function_index
        index_nu = func_2.function_index
        index_lambda = func_3.function_index
        index_sigma = func_4.function_index
        mu_center = self.basis.wfn.basisset().function_to_center(index_mu)
        nu_center = self.basis.wfn.basisset().function_to_center(index_nu)
        lambda_center = self.basis.wfn.basisset().function_to_center(index_lambda)
        sigma_center = self.basis.wfn.basisset().function_to_center(index_sigma)

        centers = np.asarray([mu_center, nu_center, lambda_center, sigma_center])
        exp_index = np.where(centers == atom)[0]

        #Form the required integrals in each dimension (am_ints is the final variable)
        am_1 = func_1.am
        am_2 = func_2.am
        am_3 = func_3.am
        am_4 = func_4.am

        am_pair = np.asarray([am_1, am_2]).astype(int)
        am_pair2 = np.asarray([am_3, am_4]).astype(int)

        max_am = np.max(am_pair)
        am_values = np.arange(max_am+1)
        am_grid = np.asarray(np.meshgrid(am_values, am_values, am_values)).T.reshape(-1, 3)
        am_ints1 = am_grid[np.sum(am_grid, axis=1) == am_pair[0]]
        am_ints2 = am_grid[np.sum(am_grid, axis=1) == am_pair[1]]
        am_1 = np.tile(am_ints1, (am_ints2.shape[0], 1))
        am_2 = np.repeat(am_ints2, am_ints1.shape[0], axis=0)
        am_ints_uv = np.hstack((am_1, am_2))
        am_ints_uv = am_ints_uv.reshape(am_ints_uv.shape[0], 2, -1)

        am_ints_uv = self.basis.reorder_int(am_ints_uv, am_pair)

        max_am = np.max(am_pair2)
        am_values = np.arange(max_am+1)
        am_grid = np.asarray(np.meshgrid(am_values, am_values, am_values)).T.reshape(-1, 3)
        am_ints1 = am_grid[np.sum(am_grid, axis=1) == am_pair2[0]]
        am_ints2 = am_grid[np.sum(am_grid, axis=1) == am_pair2[1]]
        am_1 = np.tile(am_ints1, (am_ints2.shape[0], 1))
        am_2 = np.repeat(am_ints2, am_ints1.shape[0], axis=0)
        am_ints_ls = np.hstack((am_1, am_2))
        am_ints_ls = am_ints_ls.reshape(am_ints_ls.shape[0], 2, -1)

        am_ints_ls = self.basis.reorder_int(am_ints_ls, am_pair2)

        am_ints = np.asarray(list(it.product(am_ints_uv, am_ints_ls))).reshape(-1, 4, 3)

        coefs, exp_list, RPQ_coords, PA_coords = self.calculate_primitives(func_1,
                func_2,
                func_3,
                func_4,
                xyz_mu,
                xyz_nu,
                xyz_lambda,
                xyz_sigma
                )

        coefs = np.asarray(coefs)
        exp_list = np.asarray(exp_list)
        RPQ_coords = np.asarray(RPQ_coords)
        PA_coords = np.asarray(PA_coords)

        integral_total = np.zeros((am_ints.shape[0]))

        for i in range(am_ints.shape[0]):
            ERI_dr = 0 
            for e_index in exp_index:
                am_dr = np.zeros_like(am_ints[i])
                am_dr += am_ints[i]
                am_dr[e_index, component] += 1

                am_g_x = am_dr[:, 0].sum()
                am_g_y = am_dr[:, 1].sum()
                am_g_z = am_dr[:, 2].sum()
                am_g_integrals = {}

                am_set = list(it.product(list(range(am_g_x+1)), list(range(am_g_y+1)), list(range(am_g_z+1))))
                for integral in am_set:
                    theta = self.recursive_theta_i(
                            xyz_mu,
                            xyz_nu,
                            xyz_lambda,
                            xyz_sigma,
                            RPQ_coords,
                            PA_coords,
                            exp_list,
                            integral[0],
                            integral[1],
                            integral[2],
                            0.0
                            )

                    am_g_integrals[integral] = theta

                am_ef_integrals = {}

                am_e_x = am_dr[0:2, 0].sum()
                am_e_y = am_dr[0:2, 1].sum()
                am_e_z = am_dr[0:2, 2].sum()
                am_f_x = am_dr[2:, 0].sum()
                am_f_y = am_dr[2:, 1].sum()
                am_f_z = am_dr[2:, 2].sum()
                if np.asarray([am_f_x, am_f_y, am_f_z]).any():
                    am_set_f = list(it.product(list(range(am_f_x+1)), list(range(am_f_y+1)), list(range(am_f_z+1))))
                    am_set_e = list(it.product(list(range(am_e_x+1)), list(range(am_e_y+1)), list(range(am_e_z+1))))
                    am_set_ef = list(it.product(am_set_e, am_set_f))

                    for integral in am_set_ef:
                        theta = self.recursive_electron_transfer(
                            am_g_integrals,
                            xyz_mu,
                            xyz_nu,
                            xyz_lambda,
                            xyz_sigma,
                            RPQ_coords,
                            exp_list,
                            integral[0][0],
                            integral[0][1],
                            integral[0][2],
                            integral[1][0],
                            integral[1][1],
                            integral[1][2],
                            0.0
                            )

                        theta *= coefs
                        theta *= 2.0 * exp_list[:, e_index]
                        key = tuple((integral[0][0], integral[0][1], integral[0][2], integral[1][0], integral[1][1], integral[1][2]))
                        am_ef_integrals[key] = theta.sum()
                        #am_ef_integrals[key] = theta
    
                else:
                    am_int_tmp = {}
                    for key, val in am_g_integrals.items():
                        new_key = tuple((key[0], key[1], key[2], 0, 0, 0))
                        new_val = val * coefs
                        new_val *= 2.0 * exp_list[:, e_index]
                        new_val = new_val.sum()
                        am_int_tmp[new_key] = new_val
                        #am_int_tmp[new_key] = val
    
                    am_ef_integrals = am_int_tmp

                am_i_x, am_i_y, am_i_z = am_dr[0, :]
                am_j_x, am_j_y, am_j_z = am_dr[1, :]
                am_k_x, am_k_y, am_k_z = am_dr[2, :]
                am_l_x, am_l_y, am_l_z = am_dr[3, :]
                if np.asarray([am_j_x, am_j_y, am_j_z]).any():
                    am_ijf_integrals = {}
                    am_ij_x = am_i_x + am_j_x
                    am_ij_y = am_i_y + am_j_y
                    am_ij_z = am_i_z + am_j_z
                    for key in am_ef_integrals.keys():
                        if key[0] == am_ij_x or key[1] == am_ij_y or key[2] == am_ij_z:
                            theta = self.recursive_horizontal_transfer_j(am_ef_integrals,
                                                                         xyz_mu,
                                                                         xyz_nu,
                                                                         am_i_x,
                                                                         am_i_y,
                                                                         am_i_z,
                                                                         am_j_x,
                                                                         am_j_y,
                                                                         am_j_z,
                                                                         key[3],
                                                                         key[4],
                                                                         key[5])
    
                            key = tuple((am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z, key[3], key[4], key[5]))
                            am_ijf_integrals[key] = theta

                else:
                    am_int_tmp = {}
                    for key, val in am_ef_integrals.items():
                        new_key = tuple((key[0], key[1], key[2], 0, 0, 0, key[3], key[4], key[5]))
                        am_int_tmp[new_key] = val
                    am_ijf_integrals = am_int_tmp
    
                am_ijkl_integrals = {}
                if np.asarray([am_l_x, am_l_y, am_l_z]).any():
                    theta = self.recursive_horizontal_transfer_l(am_ijf_integrals,
                                                                 xyz_lambda,
                                                                 xyz_sigma,
                                                                 am_i_x,
                                                                 am_i_y,
                                                                 am_i_z,
                                                                 am_j_x,
                                                                 am_j_y,
                                                                 am_j_z,
                                                                 am_k_x,
                                                                 am_k_y,
                                                                 am_k_z,
                                                                 am_l_x,
                                                                 am_l_y,
                                                                 am_l_z
                                                                 )
                    
                    key = tuple((am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z,
                                 am_k_x, am_k_y, am_k_z, am_l_x, am_l_y, am_l_z))
                    am_ijkl_integrals[key] = theta

                else:
                    am_int_tmp = {}
                    for key, val in am_ijf_integrals.items():
                        new_key = tuple((key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7], key[8], 0, 0, 0))
                        am_int_tmp[new_key] = val
                    am_ijkl_integrals = am_int_tmp
    
                key = tuple((am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z,
                         am_k_x, am_k_y, am_k_z, am_l_x, am_l_y, am_l_z))

                term1 = am_ijkl_integrals[key]
                
                am_dr = np.zeros_like(am_ints[i])
                am_dr += am_ints[i]
                am_dr[e_index, component] -= 1
                
                am_g_x = am_dr[:, 0].sum()
                am_g_y = am_dr[:, 1].sum()
                am_g_z = am_dr[:, 2].sum()
                am_g_integrals = {}

                am_set = list(it.product(list(range(am_g_x+1)), list(range(am_g_y+1)), list(range(am_g_z+1))))
                for integral in am_set:
                    theta = self.recursive_theta_i(
                            xyz_mu,
                            xyz_nu,
                            xyz_lambda,
                            xyz_sigma,
                            RPQ_coords,
                            PA_coords,
                            exp_list,
                            integral[0],
                            integral[1],
                            integral[2],
                            0.0
                            )

                    am_g_integrals[integral] = theta

                am_ef_integrals = {}

                am_e_x = am_dr[0:2, 0].sum()
                am_e_y = am_dr[0:2, 1].sum()
                am_e_z = am_dr[0:2, 2].sum()
                am_f_x = am_dr[2:, 0].sum()
                am_f_y = am_dr[2:, 1].sum()
                am_f_z = am_dr[2:, 2].sum()

                am_f = np.asarray([am_f_x, am_f_y, am_f_z])
                if am_f.any() and not len(np.where(am_dr < 0)[0]):
                    am_set_f = list(it.product(list(range(am_f_x+1)), list(range(am_f_y+1)), list(range(am_f_z+1))))
                    am_set_e = list(it.product(list(range(am_e_x+1)), list(range(am_e_y+1)), list(range(am_e_z+1))))
                    am_set_ef = list(it.product(am_set_e, am_set_f))

                    for integral in am_set_ef:
                        theta = self.recursive_electron_transfer(
                            am_g_integrals,
                            xyz_mu,
                            xyz_nu,
                            xyz_lambda,
                            xyz_sigma,
                            RPQ_coords,
                            exp_list,
                            integral[0][0],
                            integral[0][1],
                            integral[0][2],
                            integral[1][0],
                            integral[1][1],
                            integral[1][2],
                            0.0
                            )

                        theta *= coefs
                        key = tuple((integral[0][0], integral[0][1], integral[0][2], integral[1][0], integral[1][1], integral[1][2]))
                        am_ef_integrals[key] = theta.sum()
                        #am_ef_integrals[key] = theta

                else:

                    am_int_tmp = {}
                    for key, val in am_g_integrals.items():
                        new_key = tuple((key[0], key[1], key[2], 0, 0, 0))
                        new_val = val * coefs
                        new_val = new_val.sum()
                        am_int_tmp[new_key] = new_val
                        #am_int_tmp[new_key] = val

                    am_ef_integrals = am_int_tmp

                am_i_x, am_i_y, am_i_z = am_dr[0, :]
                am_j_x, am_j_y, am_j_z = am_dr[1, :]
                am_k_x, am_k_y, am_k_z = am_dr[2, :]
                am_l_x, am_l_y, am_l_z = am_dr[3, :]

                am_j = np.asarray([am_j_x, am_j_y, am_j_z])
                if am_j.any() and not len(np.where(am_dr < 0)[0]):
                    am_ijf_integrals = {}
                    am_ij_x = am_i_x + am_j_x
                    am_ij_y = am_i_y + am_j_y
                    am_ij_z = am_i_z + am_j_z
                    for key in am_ef_integrals.keys():
                        if key[0] == am_ij_x or key[1] == am_ij_y or key[2] == am_ij_z:
                            theta = self.recursive_horizontal_transfer_j(am_ef_integrals,
                                                                         xyz_mu,
                                                                         xyz_nu,
                                                                         am_i_x,
                                                                         am_i_y,
                                                                         am_i_z,
                                                                         am_j_x,
                                                                         am_j_y,
                                                                         am_j_z,
                                                                         key[3],
                                                                         key[4],
                                                                         key[5])

                            key = tuple((am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z, key[3], key[4], key[5]))
                            am_ijf_integrals[key] = theta

                else:
                    am_int_tmp = {}
                    for key, val in am_ef_integrals.items():
                        new_key = tuple((key[0], key[1], key[2], 0, 0, 0, key[3], key[4], key[5]))
                        am_int_tmp[new_key] = val
                    am_ijf_integrals = am_int_tmp

                am_ijkl_integrals = {}

                am_l = np.asarray([am_l_x, am_l_y, am_l_z])
                if am_l.any() and not len(np.where(am_dr < 0)[0]):
                    theta = self.recursive_horizontal_transfer_l(am_ijf_integrals,
                                                                 xyz_lambda,
                                                                 xyz_sigma,
                                                                 am_i_x,
                                                                 am_i_y,
                                                                 am_i_z,
                                                                 am_j_x,
                                                                 am_j_y,
                                                                 am_j_z,
                                                                 am_k_x,
                                                                 am_k_y,
                                                                 am_k_z,
                                                                 am_l_x,
                                                                 am_l_y,
                                                                 am_l_z
                                                                 )

                    key = tuple((am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z,
                                 am_k_x, am_k_y, am_k_z, am_l_x, am_l_y, am_l_z))
                    am_ijkl_integrals[key] = theta

                else:
                    am_int_tmp = {}
                    for key, val in am_ijf_integrals.items():
                        new_key = tuple((key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7], key[8], 0, 0, 0))
                        am_int_tmp[new_key] = val
                    am_ijkl_integrals = am_int_tmp

                key = tuple((am_i_x, am_i_y, am_i_z, am_j_x, am_j_y, am_j_z,
                         am_k_x, am_k_y, am_k_z, am_l_x, am_l_y, am_l_z))
                
                if -1 in key:
                    term2 = 0.0
                else:
                    term2 = am_ijkl_integrals[key]
               
                #ERI_dr += (2.0 * exp_list[:, e_index] * term1 - am_ints[i, e_index, component] * term2)
                ERI_dr += (term1 - am_ints[i, e_index, component] * term2)
            
            #ERI_dr *= coefs
            #integral_total[i] = ERI_dr.sum()
            integral_total[i] = ERI_dr

        return integral_total

    def calculate(self):
        """
        Calculate ERI 
        """

        two_shell_pair = self._make_two_shell_pair()
        for p_num, p in enumerate(two_shell_pair):
            func_1 = p[0]
            func_2 = p[1]
            func_3 = p[2]
            func_4 = p[3]

            num_mu = func_1.nfunction
            index_mu = func_1.function_index
            num_nu = func_2.nfunction
            index_nu = func_2.function_index
            num_lambda = func_3.nfunction
            index_lambda = func_3.function_index
            num_sigma = func_4.nfunction
            index_sigma = func_4.function_index

            mu_center = self.basis.wfn.basisset().function_to_center(index_mu)
            nu_center = self.basis.wfn.basisset().function_to_center(index_nu)
            lambda_center = self.basis.wfn.basisset().function_to_center(index_lambda)
            sigma_center = self.basis.wfn.basisset().function_to_center(index_sigma)

            xyz_mu = self.basis.get_center_position(mu_center)
            xyz_nu = self.basis.get_center_position(nu_center)
            xyz_lambda = self.basis.get_center_position(lambda_center)
            xyz_sigma = self.basis.get_center_position(sigma_center)

            integrals = self.calculate_integral(func_1, func_2, func_3, func_4, 
                    xyz_mu, xyz_nu, xyz_lambda, xyz_sigma)
            
            index = 0
            for i in range(index_mu, index_mu+num_mu):
                for j in range(index_nu, index_nu+num_nu):
                    for k in range(index_lambda, index_lambda+num_lambda):
                        for l in range(index_sigma, index_sigma+num_sigma):
                            self.ERI[i, j, k, l] = self.ERI[k, l, i, j] = self.ERI[j, i, l, k] = self.ERI[l, k, j, i] \
                                    = self.ERI[j, i, k, l] = self.ERI[l, k, i, j] = self.ERI[i, j, l, k] = self.ERI[k, l, j, i] = integrals[index]
                            index += 1
        
        if self.basis.spherical:
            #Transform all axes from cartesian to spherical using einsum
            c2sph = self.basis.c2sph
            self.ERI = np.einsum('ij, jklm->iklm', c2sph.T, self.ERI)
            self.ERI = np.einsum('ik, jklm->jilm', c2sph.T, self.ERI)
            self.ERI = np.einsum('ijkl, km->ijml', self.ERI, c2sph)
            self.ERI = np.einsum('ijkl, lm->ijkm', self.ERI, c2sph)
        return self.ERI

    def calculate_gradient(self, atom, component):
        """
        Calculate ERI gradient with respect to nucleus and component
        """
        ERI_dr = np.zeros((self.basis.num_func, self.basis.num_func, self.basis.num_func, self.basis.num_func))
        two_shell_pair = self._make_two_shell_pair()
        for p_num, p in enumerate(two_shell_pair):
            func_1 = p[0]
            func_2 = p[1]
            func_3 = p[2]
            func_4 = p[3]

            num_mu = func_1.nfunction
            index_mu = func_1.function_index
            num_nu = func_2.nfunction
            index_nu = func_2.function_index
            num_lambda = func_3.nfunction
            index_lambda = func_3.function_index
            num_sigma = func_4.nfunction
            index_sigma = func_4.function_index

            mu_center = self.basis.wfn.basisset().function_to_center(index_mu)
            nu_center = self.basis.wfn.basisset().function_to_center(index_nu)
            lambda_center = self.basis.wfn.basisset().function_to_center(index_lambda)
            sigma_center = self.basis.wfn.basisset().function_to_center(index_sigma)

            xyz_mu = self.basis.get_center_position(mu_center)
            xyz_nu = self.basis.get_center_position(nu_center)
            xyz_lambda = self.basis.get_center_position(lambda_center)
            xyz_sigma = self.basis.get_center_position(sigma_center)

            integrals = self.calculate_integral_derivative(func_1, func_2, func_3, func_4,
                    xyz_mu, xyz_nu, xyz_lambda, xyz_sigma, atom, component)

            index = 0
            for i in range(index_mu, index_mu+num_mu):
                for j in range(index_nu, index_nu+num_nu):
                    for k in range(index_lambda, index_lambda+num_lambda):
                        for l in range(index_sigma, index_sigma+num_sigma):
                            ERI_dr[i, j, k, l] = ERI_dr[k, l, i, j] = ERI_dr[j, i, l, k] = ERI_dr[l, k, j, i] \
                                    = ERI_dr[j, i, k, l] = ERI_dr[l, k, i, j] = ERI_dr[i, j, l, k] = ERI_dr[k, l, j, i] = integrals[index]
                            index += 1

        if self.basis.spherical:
            #Transform all axes from cartesian to spherical using einsum
            c2sph = self.basis.c2sph
            ERI_dr = np.einsum('ij, jklm->iklm', c2sph.T, ERI_dr)
            ERI_dr = np.einsum('ik, jklm->jilm', c2sph.T, ERI_dr)
            ERI_dr = np.einsum('ijkl, km->ijml', ERI_dr, c2sph)
            ERI_dr = np.einsum('ijkl, lm->ijkm', ERI_dr, c2sph)
        return ERI_dr

