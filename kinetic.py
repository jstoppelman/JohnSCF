import numpy as np
import sys

class Kinetic:
    """
    Compute Kinetic Energy Matrix
    """
    def __init__(self, basis):
        """
        Parameters
        -----------
        basis: class
            Class containing Psi4 basis info
        """
        self.basis = basis
        self.T = np.zeros((self.basis.num_func, self.basis.num_func))

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

        Returns
        -----------
        s00: np.ndarray
            standard Gaussian overlap integral in each dimension
        coef: np.ndarray
            Product of basis function coefficients
        """
        
        coef = coef1 * coef2
        alpha = exp1 * exp2
        exp_sum_inv = 1/(exp1+exp2)
        R2 = ((xyz1-xyz2)**2)

        #Prefactor for Gaussian integral
        pref = (np.pi * exp_sum_inv)**(1/2)
        s00 = pref * np.exp(-alpha * exp_sum_inv * R2)

        return s00, coef

    def get_t00_integral(self, exp1, exp_sum_inv, PA, s00):
        """
        Calculate the basic kinetic energy integral

        Parameters
        -----------
        exp1: float
            Basis function exponent 1
        exp_sum_inv: float
            Basis function exponent sum
        PA: array
            P coordinate from first basis function
        s00: float
            Basic overlap integral

        Returns
        -----------
        t00: float
            Kinetic energy integral base integral
        """
        t00 = (exp1 - 2 * exp1**2 * (PA**2 + 0.5 * exp_sum_inv)) * s00
        return t00

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
        Recursion for overlap integrals

        Parameters
        -----------
        PA: np.ndarray
            XPA in a given dimension for all basis functions
        PB: np.ndarray
            XPB in a given dimension for all basis functions
        exp_sum: np.ndarray
            List of the inverse sum for each exponent
        am_i: int
            angular momentum corresponding to first integral component
        am_j: int
            angular momentum corresponding to second integral component
        cart_fac: np.ndarray
            Primitive Gaussian overlap integral

        Returns
        -----------
        np.ndarray
            Integral values for all basis functions after performing recursion
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

    def obara_saika_recursion_T(self, PA, PB, exp_list, exp_sum, am_i, am_j, S00, T00):
        """
        Recursion for kinetic energy integrals

        Parameters
        -----------
        PA: np.ndarray
            XPA in a given dimension for all basis functions
        PB: np.ndarray
            XPB in a given dimension for all basis functions
        exp_list: np.ndarray
            List of exponents
        exp_sum: np.ndarray
            List of the inverse sum for each exponent
        am_i: int
            angular momentum corresponding to first integral component
        am_j: int
            angular momentum corresponding to second integral component
        S00: np.ndarray
            Primitive Gaussian overlap integral
        T00: np.ndarray
            Primitive Gaussian kinetic energy integral

        Returns
        -----------
        np.ndarray
            Integral values for all basis functions after performing recursion
        """
        if am_i <= 0.0 and am_j <= 0.0:
            return T00

        if am_j > am_i:
            return (PB * self.obara_saika_recursion_T(PA, PB, exp_list, exp_sum, am_i, am_j-1, S00, T00)) + \
                    0.5 * exp_sum * (am_i * self.obara_saika_recursion_T(PA, PB, exp_list, exp_sum, am_i-1, am_j-1, S00, T00) + \
                    (am_j-1) * self.obara_saika_recursion_T(PA, PB, exp_list, exp_sum, am_i, am_j-2, S00, T00)) + \
                    exp_list[:, 0] * exp_sum * (2 * exp_list[:, 1] * self.obara_saika_recursion(PA, PB, exp_sum, am_i, am_j, S00) - \
                    (am_j-1) * self.obara_saika_recursion(PA, PB, exp_sum, am_i, am_j-2, S00))
        else:
            return (PA * self.obara_saika_recursion_T(PA, PB, exp_list, exp_sum, am_i-1, am_j, S00, T00)) + \
                    0.5 * exp_sum * ((am_i-1) * self.obara_saika_recursion_T(PA, PB, exp_list, exp_sum, am_i-2, am_j, S00, T00) + \
                    (am_j) * self.obara_saika_recursion_T(PA, PB, exp_list, exp_sum, am_i-1, am_j-1, S00, T00)) + \
                    exp_list[:, 1] * exp_sum * (2 * exp_list[:, 0] * self.obara_saika_recursion(PA, PB, exp_sum, am_i, am_j, S00) - \
                    (am_i-1) * self.obara_saika_recursion(PA, PB, exp_sum, am_i-2, am_j, S00))

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
        exp_list = []
        exp_sum_inv_list = []
        s00_ints = []
        coefs = []
        t00_ints = []
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

                s00, coef = self.get_s00_integral(coef_mu, coef_nu, exp_mu, exp_nu, xyz_mu, xyz_nu)
                s00_ints.append(s00)
                coefs.append(coef)
                
                t00_int = []
                for cart in range(3):
                    t00 = self.get_t00_integral(exp_mu, exp_sum_inv, P_coord[0][cart], s00[cart]) 
                    t00_int.append(t00)
                t00_ints.append(t00_int)

        return s00_ints, coefs, t00_ints, P_coords, exp_list, exp_sum_inv_list

    def calculate_integral(self, func_1, func_2, xyz_mu, xyz_nu):
        """
        Calculate kinetic energy integral between two basis functions

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

        #Reorder integrals to Psi4 ordering for easier comparison
        am_ints = self.basis.reorder_int(am_ints, am_pair)

        #Store integrals here
        integral_total = np.zeros((am_ints.shape[0]))

        #Get primitive integrals S00 and T00 (in 3D) as well as some other basis info
        S00_ints, coefs, T00_ints, P_coords, exp_list, exp_sum_inv_list = self.calculate_primitives(func_1, func_2, xyz_mu, xyz_nu)

        #Convert to numpy array
        S00_ints = np.asarray(S00_ints)
        coefs = np.asarray(coefs)
        T00_ints = np.asarray(T00_ints)
        exp_sum_inv_list = np.asarray(exp_sum_inv_list)
        exp_list = np.asarray(exp_list)
        P_coords = np.asarray(P_coords)

        #Not sure if this is the best way, but we need to do recursion for overlap integrals as well
        #While also doing recursion for the overlap integrals
        S_recur = [[1, 2], [0, 2], [0, 1]]
        for i in range(am_ints.shape[0]):
            #Will be (n_primitive, 3)
            #We are evaluating Tij*Skl*Smn in 3D, so do this 3 times
            Tab_ints = np.zeros_like(T00_ints)
            for cart in range(3):
                am_pair = am_ints[i, :, cart]
        
                T_comp = self.obara_saika_recursion_T(P_coords[:, 0, cart],
                                                      P_coords[:, 1, cart],
                                                      exp_list,
                                                      exp_sum_inv_list,
                                                      am_pair[0], 
                                                      am_pair[1],
                                                      S00_ints[:, cart],
                                                      T00_ints[:, cart]
                                                      )
                
                S = np.ones((T00_ints.shape[0]))
                #Compute S integrals for the other cartesian components
                for cart2 in S_recur[cart]:
                    am_pair = am_ints[i, :, cart2]
                    
                    S_comp = self.obara_saika_recursion(P_coords[:, 0, cart2],
                                                        P_coords[:, 1, cart2],
                                                        exp_sum_inv_list,
                                                        am_pair[0],
                                                        am_pair[1],
                                                        S00_ints[:, cart2]
                                                        )
                    
                    S *= S_comp
                #Store Tij*Skl*Smn for all 3 directions in Tab_ints
                Tab_ints[:, cart] = T_comp * S
          
            Tab_ints = Tab_ints.sum(axis=-1)
            Tab_ints *= coefs
            integral_total[i] = Tab_ints.sum()
         
        return integral_total

    def calculate_integral_derivative(self, func_1, func_2, xyz_mu, xyz_nu, atom, component):
        """
        Calculate kinetic energy integral derivative between two basis functions

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
        integral = 0.0

        index_mu = func_1.function_index
        index_nu = func_2.function_index
        mu_center = self.basis.wfn.basisset().function_to_center(index_mu)
        nu_center = self.basis.wfn.basisset().function_to_center(index_nu)

        centers = np.asarray([mu_center, nu_center])
        exp_index = np.where(centers == atom)[0][0]

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

        #Reorder integrals to Psi4 ordering for easier comparison
        am_ints = self.basis.reorder_int(am_ints, am_pair)

        integral_total = np.zeros((am_ints.shape[0]))

        #Get primitive integrals S00 and T00 (in 3D) as well as some other basis info
        S00_ints, coefs, T00_ints, P_coords, exp_list, exp_sum_inv_list = self.calculate_primitives(func_1, func_2, xyz_mu, xyz_nu)

        #Convert to numpy array
        S00_ints = np.asarray(S00_ints)
        coefs = np.asarray(coefs)
        T00_ints = np.asarray(T00_ints)
        exp_sum_inv_list = np.asarray(exp_sum_inv_list)
        exp_list = np.asarray(exp_list)
        P_coords = np.asarray(P_coords)

        a = exp_list[:, exp_index]

        #Not sure if this is the best way, but we need to do recursion for overlap integrals as well
        #While also doing recursion for the overlap integrals
        S_recur = [[1, 2], [0, 2], [0, 1]]
        for i in range(am_ints.shape[0]):
            #Will be (n_primitive, 3)
            #We are evaluating Tij*Skl*Smn in 3D, so do this 3 times
            Tab_ints = np.zeros_like(T00_ints)
            for cart in range(3):
    
                am_pair = am_ints[i, :, cart]
                
                if cart == component:
                    am_dr = np.zeros_like(am_pair)
                    am_dr += am_pair
                    am_dr[exp_index] += 1

                    term_1 = self.obara_saika_recursion_T(P_coords[:, 0, cart],
                                                          P_coords[:, 1, cart],
                                                          exp_list,
                                                          exp_sum_inv_list,
                                                          am_dr[0],
                                                          am_dr[1],
                                                          S00_ints[:, cart],
                                                          T00_ints[:, cart]
                                                          )

                    am_dr = np.zeros_like(am_pair)
                    am_dr += am_pair
                    am_dr[exp_index] -= 1
                    term_2 = self.obara_saika_recursion_T(P_coords[:, 0, cart],
                                                          P_coords[:, 1, cart],
                                                          exp_list,
                                                          exp_sum_inv_list,
                                                          am_dr[0],
                                                          am_dr[1],
                                                          S00_ints[:, cart],
                                                          T00_ints[:, cart]
                                                          )

                    T_comp = (2.0 * a * term_1 - am_pair[exp_index] * term_2)
                
                else:
                    T_comp = self.obara_saika_recursion_T(P_coords[:, 0, cart],
                                                          P_coords[:, 1, cart],
                                                          exp_list,
                                                          exp_sum_inv_list,
                                                          am_pair[0],
                                                          am_pair[1],
                                                          S00_ints[:, cart],
                                                          T00_ints[:, cart]
                                                          )

                S = np.ones((T00_ints.shape[0]))
                #Compute S integrals for the other cartesian components
                for cart2 in S_recur[cart]:
                    am_pair = am_ints[i, :, cart2]

                    if cart2 == component:
                        am_dr = np.zeros_like(am_pair)
                        am_dr += am_pair
                        am_dr[exp_index] += 1

                        term_1 = self.obara_saika_recursion(P_coords[:, 0, cart2],
                                                            P_coords[:, 1, cart2],
                                                            exp_sum_inv_list,
                                                            am_dr[0],
                                                            am_dr[1],
                                                            S00_ints[:, cart2]
                                                            )

                        am_dr = np.zeros_like(am_pair)
                        am_dr += am_pair
                        am_dr[exp_index] -= 1

                        term_2 = self.obara_saika_recursion(P_coords[:, 0, cart2],
                                                            P_coords[:, 1, cart2],
                                                            exp_sum_inv_list,
                                                            am_dr[0],
                                                            am_dr[1],
                                                            S00_ints[:, cart2]
                                                            )

                        S_comp = (2.0 * a * term_1 - am_pair[exp_index] * term_2)
                    else:

                        S_comp = self.obara_saika_recursion(P_coords[:, 0, cart2],
                                                            P_coords[:, 1, cart2],
                                                            exp_sum_inv_list,
                                                            am_pair[0],
                                                            am_pair[1],
                                                            S00_ints[:, cart2]
                                                            )

                    S *= S_comp
                #Store Tij*Skl*Smn for all 3 directions in Tab_ints
                Tab_ints[:, cart] = T_comp * S

            Tab_ints = Tab_ints.sum(axis=-1)
            Tab_ints *= coefs   
            integral_total[i] = Tab_ints.sum()     

        return integral_total

    def calculate(self):
        """
        Calculate Kinetic Energy matrix
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
                    self.T[i, j] = self.T[j, i] = integrals[index]
                    index += 1
       
        if self.basis.spherical:
            c2sph = self.basis.c2sph     
            #Transform to spherical basis (similar to PySCF)
            self.T = c2sph.T.dot(self.T).dot(c2sph)   
        return self.T

    def calculate_gradient(self, atom, component):
        """
        Calculate kinetic energy matrix gradient in AO basis
        """
        
        T_dr = np.zeros((self.basis.num_func, self.basis.num_func))
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
                        T_dr[i, j] = T_dr[j, i] = integrals[index]
                        index += 1

        if self.basis.spherical:
            c2sph = self.basis.c2sph
            T_dr = c2sph.T.dot(T_dr).dot(c2sph)
        return T_dr
