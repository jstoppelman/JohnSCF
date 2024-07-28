import numpy as np
import sys

class Overlap:
    """
    Compute Overlap Matrix
    """
    def __init__(self, basis):
        """
        Parameters
        -----------
        basis: class
            Class containing Psi4 basis info
        """
        self.basis = basis
        self.S = np.zeros((self.basis.num_func, self.basis.num_func))

    def get_s00_integral(self, coef1, coef2, exp1, exp2, xyz1, xyz2):
        """
        Calculate standard Gaussian integral

        Parameters
        -----------
        coef1: float
            Basis function coefficient 1
        coef2: float
            Basis function coefficient 2
        exp1: float
            Basis function exponent 1
        exp2: float
            Basis function exponent 2
        xyz1: array
            size 3 array containing atomic positions corresponding to first basis function
        xyz2: array
            size 3 array containing atomic positions corresponding to second basis function
        """
        coef = coef1 * coef2
        alpha = exp1 * exp2
        exp_sum_inv = 1/(exp1+exp2)
        R2 = ((xyz1 - xyz2)**2).sum(axis=0)
        
        #Prefactor for Gaussian integral
        pref = (np.pi * exp_sum_inv)**(3/2)
        integ = pref * np.exp(-alpha * exp_sum_inv * R2)
        integ *= coef
        return integ

    def calculate_P_coords(self, exp_mu, exp_nu, xyz_mu, xyz_nu):
        """
        Return the P coordinate for computing overlap integrals

        Paramters
        ----------
        exp_mu: np.array
            Exponent for first gaussian
        exp_nu: np.array
            Exponent for second gaussian
        xyz_mu: np.array size 3
            coordinates of first center
        xyz_nu: np.array size 3
            coordinates of second center

        Returns
        ----------
        P_coord: list
            List of PXA, PXB, PYA etc.
        """
        P_xyz = (exp_mu * xyz_mu + exp_nu * xyz_nu) * 1/(exp_mu + exp_nu)
        PA = P_xyz - xyz_mu
        PB = P_xyz - xyz_nu
        return [PA, PB]

    def obara_saika_recursion(self, PA, PB, exp_sum, am_i, am_j, cart_fac):
        """
        Run Obara-Saika recursion

        Parameters
        -----------
        PA: np.ndarray
            Vector from first Gaussian center to P coordinate value
        PB: np.ndarray
            Vector from second Gaussian center to P coordinate value
        exp_sum: np.ndarray
            1/p, inverse sum of exponents
        am_i: int
            Angular momentum value associated with first orbital
        am_j: int
            Angular momentum value associated with second orbital
        """
        if am_i <= 0.0 and am_j <= 0.0:
            return cart_fac

        if am_j > am_i:
            return (PB * self.obara_saika_recursion(PA, PB, exp_sum, am_i, am_j-1, cart_fac)) + \
                0.5 * exp_sum * (am_i * self.obara_saika_recursion(PA, PB, exp_sum, am_i-1, am_j-1, cart_fac) + \
                (am_j-1) * self.obara_saika_recursion(PA, PB, exp_sum, am_i, am_j-2, cart_fac))
        else:
            return (PA * self.obara_saika_recursion(PA, PB, exp_sum, am_i-1, am_j, cart_fac)) + \
                0.5 * exp_sum * ((am_i-1) * self.obara_saika_recursion(PA, PB, exp_sum, am_i-2, am_j, cart_fac) + \
                (am_j) * self.obara_saika_recursion(PA, PB, exp_sum, am_i-1, am_j-1, cart_fac))
    
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
        integral_primitive: list
            Individual Gaussian primitive integrals
        P_coords: list
            Needed for calculating higher momentum Gaussian integrals
        exp_list: list
            exponents for both integrals involved in Gaussian
        exp_sum_inv_list: list
            Sum of exponents for each integral
        """
        integral_primitive = []        
        P_coords = []                 
        exp_list = []
        exp_sum_inv_list = [] 
        for p1 in range(func_1.nprimitive):    
            for p2 in range(func_2.nprimitive):   
                coef_mu = func_1.coef(p1)    
                coef_nu = func_2.coef(p2)   
                exp_mu = func_1.exp(p1)      
                exp_nu = func_2.exp(p2)    
                exp_list.append([exp_mu, exp_nu])
                exp_sum_inv = 1/(exp_mu + exp_nu)      
                exp_sum_inv_list.append(exp_sum_inv)     
                
                P_coord = self.calculate_P_coords(exp_mu, exp_nu, xyz_mu, xyz_nu)        
                P_coords.append(P_coord)        

                s00 = self.get_s00_integral(coef_mu, coef_nu, exp_mu, exp_nu, xyz_mu, xyz_nu)       
                integral_primitive.append(s00)

        return integral_primitive, P_coords, exp_list, exp_sum_inv_list

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

        am_ints = self.basis.reorder_int(am_ints, am_pair)
        
        integral_total = np.zeros((am_ints.shape[0]))

        integral_primitive, P_coords, exp_list, exp_sum_inv_list = self.calculate_primitives(func_1, func_2, xyz_mu, xyz_nu)

        exp_sum_inv_list = np.asarray(exp_sum_inv_list)
        P_coords = np.asarray(P_coords)
        
        #S00 integrals
        integral_primitive = np.asarray(integral_primitive)
        
        for i in range(am_ints.shape[0]):
            fac = np.ones_like(integral_primitive)
            for cart in range(3):
                am_pair = am_ints[i, :, cart]
                cart_fac = np.ones_like(integral_primitive)
                cart_fac = self.obara_saika_recursion(P_coords[:, 0, cart],
                        P_coords[:, 1, cart],
                        exp_sum_inv_list,
                        am_pair[0],
                        am_pair[1],
                        cart_fac
                        )
                fac *= cart_fac
            
            integral_add = integral_primitive * fac
            integral_add = integral_add.sum(axis=-1)
            integral_total[i] = integral_add
        
        return integral_total

    def calculate_integral_derivative(self, func_1, func_2, xyz_mu, xyz_nu, atom, component):
        """
        Calculate overlap integral derivative between two basis functions

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
        atom: int
            Index of atom to compute derivative for
        component: int
            Index of cartesian component to compute derivative for

        Returns
        -----------
        integral: array
            Sum of 3D Gaussian integrals over primitives for the basis function
        """
        cartesian = [[1, 2], [0, 2], [0, 1]]

        index_mu = func_1.function_index
        index_nu = func_2.function_index
        mu_center = self.basis.wfn.basisset().function_to_center(index_mu)
        nu_center = self.basis.wfn.basisset().function_to_center(index_nu)
        
        centers = np.asarray([mu_center, nu_center])
        exp_index = np.where(centers == atom)[0][-1]

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

        am_ints = self.basis.reorder_int(am_ints, am_pair)

        integral_total = np.zeros((am_ints.shape[0]))

        integral_primitive, P_coords, exp_list, exp_sum_inv_list = self.calculate_primitives(func_1, func_2, xyz_mu, xyz_nu)
        
        exp_list = np.asarray(exp_list)
        exp_sum_inv_list = np.asarray(exp_sum_inv_list)
        P_coords = np.asarray(P_coords)

        a = exp_list[:, exp_index]

        integral_primitive = np.asarray(integral_primitive)
        
        #Compute 2*a*theta_{i+1,j} - i*theta{i-1,j}
        for i in range(am_ints.shape[0]):
            fac = np.ones_like(integral_primitive)
            for cart in cartesian[component]:
                am_pair = am_ints[i, :, cart]
                cart_fac = np.ones_like(integral_primitive)
                cart_fac = self.obara_saika_recursion(P_coords[:, 0, cart],
                        P_coords[:, 1, cart],
                        exp_sum_inv_list,
                        am_pair[0],
                        am_pair[1],
                        cart_fac
                        )
                fac *= cart_fac

            am_pair = am_ints[i, :, component]
            
            am_dr = np.zeros_like(am_pair)
            am_dr += am_pair
            am_dr[exp_index] += 1

            cart_fac = np.ones_like(integral_primitive)
            term_1 = self.obara_saika_recursion(P_coords[:, 0, component],
                                                P_coords[:, 1, component],
                                                exp_sum_inv_list,
                                                am_dr[0],
                                                am_dr[1],
                                                cart_fac
                                                )

            am_dr = np.zeros_like(am_pair)
            am_dr += am_pair
            am_dr[exp_index] -= 1
            cart_fac = np.ones_like(integral_primitive)
            term_2 = self.obara_saika_recursion(P_coords[:, 0, component],
                                                P_coords[:, 1, component],
                                                exp_sum_inv_list,
                                                am_dr[0],
                                                am_dr[1],
                                                cart_fac
                                                )

            fac *= (2.0 * a * term_1 - am_pair[exp_index] * term_2)

            integral_add = integral_primitive * fac
            integral_add = integral_add.sum(axis=-1)
            integral_total[i] = integral_add
        
        return integral_total

    def calculate(self):
        """
        Calculate overlap matrix
        """

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

            integrals = self.calculate_integral(func_1, func_2, xyz_mu, xyz_nu)

            index = 0
            for i in range(index_mu, index_mu+num_mu):
                for j in range(index_nu, index_nu+num_nu):
                    self.S[i, j] = self.S[j, i] = integrals[index]
                    index += 1
        
        if self.basis.spherical:
            c2sph = self.basis.c2sph
            self.S = c2sph.T.dot(self.S).dot(c2sph)
        return self.S

    def calculate_gradient(self, atom, component):
        """
        Calculate overlap matrix derivative
        """
        
        S_dr = np.zeros((self.basis.num_func, self.basis.num_func))
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

            if atom in [mu_center, nu_center] and not (atom == mu_center and atom == nu_center):

                xyz_mu = self.basis.get_center_position(mu_center)
                xyz_nu = self.basis.get_center_position(nu_center)

                integrals = self.calculate_integral_derivative(func_1, func_2, xyz_mu, xyz_nu, atom, component)

                index = 0
                for i in range(index_mu, index_mu+num_mu):
                    for j in range(index_nu, index_nu+num_nu):
                        S_dr[i, j] = S_dr[j, i] = integrals[index]
                        index += 1

        if self.basis.spherical:
            c2sph = self.basis.c2sph
            S_dr = c2sph.T.dot(S_dr).dot(c2sph)
        return S_dr
