import numpy as np
import sys
from scipy.special import hyp1f1, erf
import itertools as it

class Potential:
    """
    Compute Potential Energy Matrix
    """
    def __init__(self, basis):
        """
        Parameters
        -----------
        basis: class
            Class containing Psi4 basis info
        """
        self.basis = basis
        self.V = np.zeros((self.basis.num_func, self.basis.num_func))
        self.nuclei_R, self.nuclei_Z = self.get_nuclei()

    def get_nuclei(self):
        """
        Get positions and charge of the nuclei
        Returns
        -----------
        pos: np.ndarray
            positions
        charge: np.ndarray
            charge
        """
        pos = self.basis.pos
        charge = []
        for atom in range(self.basis.mol.natom()):
            charge.append(self.basis.mol.charge(atom))
        charge = np.asarray(charge)
        return pos, charge

    def calculate_P_coords(self, exp_mu, exp_nu, xyz_mu, xyz_nu):
        """
        Return the P coordinate (center of two Gaussians)

        Paramters
        ----------
        exp_mu: np.array
            Exponents of first integral
        exp_nu: np.array
            Exponents of second integral
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
        PB = P_xyz - xyz_nu
        PC = P_xyz[None] - self.nuclei_R
        return [PA, PB], PC

    def calculate_E0(self, RAB, exp_mu, exp_nu):
        """
        Calculate K = exp^(-mu*RAB**2)

        Parameters
        -----------
        RAB: np.array
            distance between two Gaussian centers
        exp_mu: np.array
            Exponent of first Gaussian
        exp_nu: np.array
            Exponent of second Gaussian

        Returns
        -----------
        K: np.ndarray
            coefficient value for computing potential energy matrix values
        """
        mu = (exp_mu * exp_nu)/(exp_mu + exp_nu)
        K = np.exp(-mu * RAB**2)
        return K

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
        x = (RPC**2).sum(axis=-1) * p[:, None]
        F = hyp1f1(n+0.5, n+1.5, -x) / (2 * n + 1)
        return F

    def theta(self, xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m, n, N):
        """
        Parameters
        -----------
        xyz_mu: np.ndarray
            xyz positions of center of first basis function
        xyz_nu: np.ndarray
            xyz positions of center of second basis function
        P_coords: np.ndarray
            Vector between P (weighted center of Gaussians) and A/B
        RPC_coords: np.ndarray
            Vector between P (weighted center of Gaussians) and nuclei (C)
        exp_list: np.ndarray
            Exponents of Gaussians
        exp_sum_inv_list: np.ndarray
            1/(a+b) inverse sum of exponents
        i, j, k, l, m, n: int
            Angular momentum in x, y and z directions for both basis functions
        N: int
            Order for hypergeometric function

        Returns
        ----------
        val: np.ndarray
            nuclear-electron attraction values for a given pair of basis functions and nuclei
        """
        val = np.zeros((RPC_coords.shape[0], RPC_coords.shape[1]))
        if i == j == k == l == m == n == 0:
            RAB = ((xyz_mu - xyz_nu)**2).sum(axis=0)
            mu = (np.prod(exp_list, axis=1))/(exp_list.sum(axis=1))
            K = np.exp(-mu * RAB)
            R = self.calculate_R(RPC_coords, exp_list.sum(axis=1), N)
            val += -2.0 * np.pi * exp_sum_inv_list[:, None] * K[:, None] * R
        elif i > 0:
            val += P_coords[:, 0, 0][:, None] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i-1, j, k, l, m, n, N) + \
                    0.5 * exp_sum_inv_list[:, None] * ((i-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i-2, j, k, l, m, n, N) + \
                    j * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i-1, j-1, k, l, m, n, N)) - \
                    RPC_coords[:, :, 0] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i-1, j, k, l, m, n, N+1) - \
                    0.5 * exp_sum_inv_list[:, None] * ((i-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i-2, j, k, l, m, n, N+1) + \
                    j * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i-1, j-1, k, l, m, n, N+1))
        elif j > 0:
            val += P_coords[:, 1, 0][:, None] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j-1, k, l, m, n, N) + \
                    0.5 * exp_sum_inv_list[:, None] * (i * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i-1, j-1, k, l, m, n, N) + \
                    (j-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j-2, k, l, m, n, N)) - \
                    RPC_coords[:, :, 0] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j-1, k, l, m, n, N+1) - \
                    0.5 * exp_sum_inv_list[:, None] * (i * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i-1, j-1, k, l, m, n, N+1) + \
                    (j-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j-2, k, l, m, n, N+1))
        elif k > 0:
            val += P_coords[:, 0, 1][:, None] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k-1, l, m, n, N) + \
                    0.5 * exp_sum_inv_list[:, None] * ((k-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k-2, l, m, n, N) + \
                    l * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k-1, l-1, m, n, N)) - \
                    RPC_coords[:, :, 1] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k-1, l, m, n, N+1) - \
                    0.5 * exp_sum_inv_list[:, None] * ((k-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k-2, l, m, n, N+1) + \
                    l * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k-1, l-1, m, n, N+1))
        elif l > 0:
            val += P_coords[:, 1, 1][:, None] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l-1, m, n, N) + \
                    0.5 * exp_sum_inv_list[:, None] * (k * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k-1, l-1, m, n, N) + \
                    (l-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l-2, m, n, N)) - \
                    RPC_coords[:, :, 1] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l-1, m, n, N+1) - \
                    0.5 * exp_sum_inv_list[:, None] * (k * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k-1, l-1, m, n, N+1) + \
                    (l-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l-2, m, n, N+1))
        elif m > 0:
            val += P_coords[:, 0, 2][:, None] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m-1, n, N) + \
                    0.5 * exp_sum_inv_list[:, None] * ((m-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m-2, n, N) + \
                    n * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m-1, n-1, N)) - \
                    RPC_coords[:, :, 2] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m-1, n, N+1) - \
                    0.5 * exp_sum_inv_list[:, None] * ((m-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m-2, n, N+1) + \
                    n * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m-1, n-1, N+1))
        elif n > 0:
            val += P_coords[:, 1, 2][:, None] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m, n-1, N) + \
                    0.5 * exp_sum_inv_list[:, None] * (m * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m-1, n-1, N) + \
                    (n-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m, n-2, N)) - \
                    RPC_coords[:, :, 2] * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m, n-1, N+1) - \
                    0.5 * exp_sum_inv_list[:, None] * (m * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m-1, n-1, N+1) + \
                    (n-1) * self.theta(xyz_mu, xyz_nu, P_coords, RPC_coords, exp_list, exp_sum_inv_list, i, j, k, l, m, n-2, N+1)) 
        return val

    def grad_C(self,
            xyz_mu, 
            xyz_nu, 
            P_coords, 
            RPC_coords, 
            exp_list, 
            exp_sum_inv_list, 
            i, 
            j, 
            k, 
            l,
            m, 
            n, 
            atom,
            component):
        """
        Gradient of three center integral with respect to C. 
        Essentially the electric field from the nuclear attraction integral

        Parameters
        -----------
        xyz_mu: np.ndarray
            xyz positions of center of first basis function
        xyz_nu: np.ndarray
            xyz positions of center of second basis function
        P_coords: np.ndarray
            Vector between P (weighted center of Gaussians) and A/B
        RPC_coords: np.ndarray
            Vector between P (weighted center of Gaussians) and nuclei (C)
        exp_list: np.ndarray
            Exponents of Gaussians
        exp_sum_inv_list: np.ndarray
            1/(a+b) inverse sum of exponents
        i, j, k, l, m, n: int
            Angular momentum in x, y and z directions for both basis functions
        atom: int
            Atom to compute the derivative for
        component: int 
            Cartesian component to compute the derivative for
        
        Returns
        -----------
        val: np.ndarray
            Value of gradient of nuclear-attraction integral for pair of basis functions and the given atom
        """
       
        val = np.zeros((RPC_coords.shape[0], RPC_coords.shape[1]))

        #theta_{ij}^{m+1} This term is always needed
        term1 = self.theta(xyz_mu,
                    xyz_nu,
                    P_coords,
                    RPC_coords,
                    exp_list,
                    exp_sum_inv_list,
                    i,
                    j,
                    k,
                    l,
                    m,
                    n,
                    1
                    )

        val[:, atom] += (2.0 * exp_list.sum(axis=-1) * RPC_coords[:, atom, component] * term1[:, atom])
        if np.asarray([i, j, k, l, m, n]).any():
            if component == 0:
                if j > 0:
                    term2 = self.theta(xyz_mu, 
                        xyz_nu,
                        P_coords,
                        RPC_coords,
                        exp_list,
                        exp_sum_inv_list,
                        i,
                        j-1,
                        k,
                        l,
                        m,
                        n,
                        1)
                    val[:, atom] += term2[:, atom]
                if i > 0:
                    term2 = self.theta(xyz_mu,
                        xyz_nu,
                        P_coords,
                        RPC_coords,
                        exp_list,
                        exp_sum_inv_list,
                        i-1,
                        j,
                        k,
                        l,
                        m,
                        n,
                        1)
                    val[:, atom] += term2[:, atom]
            elif component == 1:
                if l > 0:
                    term2 = self.theta(xyz_mu,
                        xyz_nu,
                        P_coords,
                        RPC_coords,
                        exp_list,
                        exp_sum_inv_list,
                        i,
                        j,
                        k,
                        l-1,
                        m,
                        n,
                        1)
                    val[:, atom] += term2[:, atom]
                if k > 0:
                    term2 = self.theta(xyz_mu,
                        xyz_nu,
                        P_coords,
                        RPC_coords,
                        exp_list,
                        exp_sum_inv_list,
                        i,
                        j,
                        k-1,
                        l,
                        m,
                        n,
                        1)
                    val[:, atom] += term2[:, atom]
            else:
                if n > 0:
                    term2 = self.theta(xyz_mu,
                        xyz_nu,
                        P_coords,
                        RPC_coords,
                        exp_list,
                        exp_sum_inv_list,
                        i,
                        j,
                        k,
                        l,
                        m,
                        n-1,
                        1)
                    val[:, atom] += term2[:, atom]
                if m > 0:
                    term2 = self.theta(xyz_mu,
                        xyz_nu,
                        P_coords,
                        RPC_coords,
                        exp_list,
                        exp_sum_inv_list,
                        i,
                        j,
                        k,
                        l,
                        m-1,
                        n,
                        1)
                    val[:, atom] += term2[:, atom]
        
        return val

    def calculate_primitives(self, func_1, func_2, xyz_mu, xyz_nu):
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
        P_coords = []
        RPC_coords = []
        exp_list = []
        exp_sum_inv_list = []
        coefs = []
        for p1 in range(func_1.nprimitive):
            for p2 in range(func_2.nprimitive):
                coef_mu = func_1.coef(p1)
                coef_nu = func_2.coef(p2)
                exp_mu = func_1.exp(p1)
                exp_nu = func_2.exp(p2)
                exp_list.append([exp_mu, exp_nu])
                exp_sum_inv = 1/(exp_mu + exp_nu)
                exp_sum_inv_list.append(exp_sum_inv)

                P_coord, PC = self.calculate_P_coords(exp_mu, exp_nu, xyz_mu, xyz_nu)
                P_coords.append(P_coord)
                RPC_coords.append(PC)

                coef = coef_mu * coef_nu
                coefs.append(coef)

        return coefs, P_coords, RPC_coords, exp_list, exp_sum_inv_list

    def calculate_integral(self, func_1, func_2, xyz_mu, xyz_nu):
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
        integral: array
            Sum of 3D Gaussian integrals over primitives for the basis function
        """
        integral = 0.0

        #Form the required integrals in each dimension (am_ints is the final variable)
        am_1 = func_1.am
        am_2 = func_2.am
        am_pair = np.asarray([am_1, am_2]).astype(int)

        max_am = np.max(am_pair)
        am_values = np.arange(max_am+1)
        am_grid = np.asarray(np.meshgrid(am_values, am_values, am_values)).T.reshape(-1, 3)
        am_ints1 = am_grid[np.sum(am_grid, axis=1) == am_pair[0]]
        am_ints2 = am_grid[np.sum(am_grid, axis=1) == am_pair[1]]
        am_1 = np.tile(am_ints1, (am_ints2.shape[0], 1))
        am_2 = np.repeat(am_ints2, am_ints1.shape[0], axis=0)
        am_ints = np.hstack((am_1, am_2))
        am_ints = am_ints.reshape(am_ints.shape[0], 2, -1)

        coefs, P_coords, RPC_coords, exp_list, exp_sum_inv_list = self.calculate_primitives(func_1, func_2, xyz_mu, xyz_nu)

        coefs = np.asarray(coefs)
        P_coords = np.asarray(P_coords)
        RPC_coords = np.asarray(RPC_coords)
        
        exp_list = np.asarray(exp_list)
        exp_sum_inv_list = np.asarray(exp_sum_inv_list)

        #Reorder integrals to Psi4 ordering for easier comparison
        am_ints = self.basis.reorder_int(am_ints, am_pair)

        #Store integrals here
        integral_total = np.zeros((am_ints.shape[0]))
        
        for i in range(am_ints.shape[0]):

            Vab_ints = self.theta(xyz_mu, 
                    xyz_nu,
                    P_coords,
                    RPC_coords,
                    exp_list,
                    exp_sum_inv_list,
                    am_ints[i][0,0],
                    am_ints[i][1,0],
                    am_ints[i][0,1],
                    am_ints[i][1,1],
                    am_ints[i][0,2],
                    am_ints[i][1,2],
                    0
                    )
            
            Vab_ints *= self.nuclei_Z[None,:]
            Vab_ints = Vab_ints.sum(axis=-1)
            Vab_ints *= coefs
            integral_total[i] = Vab_ints.sum()

        return integral_total

    def calculate_integral_derivative(self, func_1, func_2, xyz_mu, xyz_nu, atom, component):
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
        integral: array
            Sum of 3D Gaussian integrals over primitives for the basis function
        """
        integral = 0.0

        index_mu = func_1.function_index
        index_nu = func_2.function_index
        mu_center = self.basis.wfn.basisset().function_to_center(index_mu)
        nu_center = self.basis.wfn.basisset().function_to_center(index_nu)

        centers = np.asarray([mu_center, nu_center])
        exp_index = np.where(centers == atom)[0]

        #Form the required integrals in each dimension (am_ints is the final variable)
        am_1 = func_1.am
        am_2 = func_2.am
        am_pair = np.asarray([am_1, am_2]).astype(int)

        max_am = np.max(am_pair)
        am_values = np.arange(max_am+1)
        am_grid = np.asarray(np.meshgrid(am_values, am_values, am_values)).T.reshape(-1, 3)
        am_ints1 = am_grid[np.sum(am_grid, axis=1) == am_pair[0]]
        am_ints2 = am_grid[np.sum(am_grid, axis=1) == am_pair[1]]
        am_1 = np.tile(am_ints1, (am_ints2.shape[0], 1))
        am_2 = np.repeat(am_ints2, am_ints1.shape[0], axis=0)
        am_ints = np.hstack((am_1, am_2))
        am_ints = am_ints.reshape(am_ints.shape[0], 2, -1)

        coefs, P_coords, RPC_coords, exp_list, exp_sum_inv_list = self.calculate_primitives(func_1, func_2, xyz_mu, xyz_nu)

        coefs = np.asarray(coefs)
        P_coords = np.asarray(P_coords)
        RPC_coords = np.asarray(RPC_coords)
        exp_list = np.asarray(exp_list)
        exp_sum_inv_list = np.asarray(exp_sum_inv_list)
        #Reorder integrals to Psi4 ordering for easier comparison
        am_ints = self.basis.reorder_int(am_ints, am_pair)

        #Store integrals here
        integral_total = np.zeros((am_ints.shape[0]))

        for i in range(am_ints.shape[0]):
            
            Vab_dr = np.zeros((RPC_coords.shape[0], RPC_coords.shape[1]))
            for e_index in exp_index:
                am_dr = np.zeros_like(am_ints[i])
                am_dr += am_ints[i]
                am_dr[e_index, component] += 1
                 
                term1 = self.theta(xyz_mu,
                    xyz_nu,
                    P_coords,
                    RPC_coords,
                    exp_list,
                    exp_sum_inv_list,
                    am_dr[0,0],
                    am_dr[1,0],
                    am_dr[0,1],
                    am_dr[1,1],
                    am_dr[0,2],
                    am_dr[1,2],
                    0
                    )

                am_dr = np.zeros_like(am_ints[i])
                am_dr += am_ints[i]
                am_dr[e_index, component] -= 1

                term2 = self.theta(xyz_mu,
                    xyz_nu,
                    P_coords,
                    RPC_coords,
                    exp_list,
                    exp_sum_inv_list,
                    am_dr[0,0],
                    am_dr[1,0],
                    am_dr[0,1],
                    am_dr[1,1],
                    am_dr[0,2],
                    am_dr[1,2],
                    0
                    )
                
                Vab_dr += (2.0 * exp_list[:, e_index][:, None] * term1 - am_ints[i, e_index, component] * term2)

            #Gradient with respect to C. We only need this for the atom that we are computing the derivatives for
            term3 = self.grad_C(
                    xyz_mu,
                    xyz_nu,
                    P_coords,
                    RPC_coords,
                    exp_list,
                    exp_sum_inv_list,
                    am_ints[i][0,0],
                    am_ints[i][1,0],
                    am_ints[i][0,1],
                    am_ints[i][1,1],
                    am_ints[i][0,2],
                    am_ints[i][1,2],
                    atom,
                    component
                    )


            Vab_dr += term3

            Vab_dr *= self.nuclei_Z[None,:]
            Vab_dr = Vab_dr.sum(axis=-1)
            Vab_dr *= coefs
            integral_total[i] = Vab_dr.sum()
        
        return integral_total

    def calculate(self):
        """
        Calculate Nuclear-Electron Energy matrix
        """

        shell_pairs = self.basis.shell_pairs
        #Loop through pairs
        for p in shell_pairs:
            func_1 = p[0]
            func_2 = p[1]

            num_mu = func_1.nfunction
            index_mu = func_1.function_index
            num_nu = func_2.nfunction
            index_nu = func_2.function_index
            mu_center = self.basis.wfn.basisset().function_to_center(index_mu)
            nu_center = self.basis.wfn.basisset().function_to_center(index_nu)

            xyz_mu = self.basis.get_center_position(mu_center)
            xyz_nu = self.basis.get_center_position(nu_center)

            integrals = self.calculate_integral(func_1, func_2, xyz_mu, xyz_nu)

            #Fill matrix
            index = 0
            for i in range(index_mu, index_mu+num_mu):
                for j in range(index_nu, index_nu+num_nu):
                    self.V[i, j] = self.V[j, i] = integrals[index]
                    index += 1
        
        if self.basis.spherical:
            c2sph = self.basis.c2sph     
            #Transform to spherical basis (similar to PySCF)
            self.V = c2sph.T.dot(self.V).dot(c2sph)   
        return self.V

    def calculate_gradient(self, atom, component):
        """
        Calculate electron-nuclear energy matrix gradient in AO basis
        """

        V_dr = np.zeros((self.basis.num_func, self.basis.num_func))
        shell_pairs = self.basis.shell_pairs
        for p in shell_pairs:
            func_1 = p[0]
            func_2 = p[1]
            
            num_mu = func_1.nfunction
            index_mu = func_1.function_index
            num_nu = func_2.nfunction
            index_nu = func_2.function_index
            mu_center = self.basis.wfn.basisset().function_to_center(index_mu)
            nu_center = self.basis.wfn.basisset().function_to_center(index_nu)

            xyz_mu = self.basis.get_center_position(mu_center)
            xyz_nu = self.basis.get_center_position(nu_center)

            integrals = self.calculate_integral_derivative(func_1, func_2, xyz_mu, xyz_nu, atom, component)

            index = 0
            for i in range(index_mu, index_mu+num_mu):
                for j in range(index_nu, index_nu+num_nu):
                    V_dr[i, j] = V_dr[j, i] = integrals[index]
                    index += 1

        if self.basis.spherical:
            c2sph = self.basis.c2sph
            V_dr = c2sph.T.dot(V_dr).dot(c2sph)
        return V_dr

