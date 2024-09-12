import numpy as np
import psi4
import scipy

class Basis:
    """
    Use Psi4 to read in basis set info 
    and store it in class
    """
    def __init__(self, atoms, basis, spherical):
        """
        Parameters
        -----------
        atoms: Atoms object
            contains system info
        basis: str
            Name of basis set to use
        spherical: bool
            Determines whether to use Cartesian or Spherical orbitals
        """
        self.atoms = atoms
        self.spherical = spherical

        #Should only need to input basis info here
        psi4.set_options({'basis': basis})

        #Form Psi4 mol object
        self.symbols = self.atoms.get_chemical_symbols()
        self.pos = self.atoms.get_positions()
        mol_str = ''
        for s, xyz in zip(self.symbols, self.pos):
            mol_str += f'{s}  {xyz[0]} {xyz[1]} {xyz[2]}\n'
        mol_str += 'units angstrom\n'
        self.mol = psi4.geometry(mol_str)
       
        com = self.atoms.get_center_of_mass()
        self.pos -= com
        #Convert to Bohr
        self.pos *= 1.88973

        #Make stored wavefunction
        self.wfn = psi4.core.Wavefunction.build(self.mol, psi4.core.get_global_option('BASIS'))
        
        self.shell_pairs = self._make_shell_pairs()

        self.num_func = self._get_num_func()

        self.max_am = self._get_max_am()

        self.c2sph = self.cartesian2spherical()

        #Hacky way to get integrals ordered same way as Psi4
        self.reorder = {}
        self.reorder[(0, 1)] = [1, 0, 2]
        self.reorder[(1, 0)] = [1, 0, 2]
        self.reorder[(1, 1)] = [4, 1, 7, 3, 0, 6, 5, 2, 8]
        self.reorder[(0, 2)] = [2, 1, 4, 0, 3, 5]
        self.reorder[(2, 0)] = [2, 1, 4, 0, 3, 5]
        self.reorder[(1, 2)] = [7, 4, 13, 1, 10, 16, 6, 3, 12, 0, 9, 15, 8, 5, 14, 2, 11, 17]

        self.reorder[(2, 1)] = [8, 2, 14, 7, 1, 13, 10, 4, 16, 6, 0, 12, 9, 3, 15, 11, 5, 17]
        self.reorder[(2, 2)] = [14, 13, 16, 12, 15, 17, 8, 7, 10, 6, 9, 11, 26, 25, 28, 24, 27, 29, 2, 1, 4, 0, 3, 5, 20, 19, 22, 18, 21, 23, 32, 31, 34, 30, 33, 35]

    def _make_shell_pairs(self):
        """
        Construct the necessary shell pairs for computing integrals
        """
        pairs = []
        for i in range(self.wfn.basisset().nshell()):
            shell_i = self.wfn.basisset().shell(i)
            for j in range(i, self.wfn.basisset().nshell()):
                shell_j = self.wfn.basisset().shell(j)
                pairs.append([shell_i, shell_j])
        return pairs

    def _get_num_func(self):
        """
        Get number of basis functions
        """
        num_func = 0
        for i in range(self.wfn.basisset().nshell()):
            shell = self.wfn.basisset().shell(i)
            for f in range(shell.nfunction):
                num_func += 1
        return num_func

    def _get_max_am(self):
        """
        Get maximum angular momentum in basis
        """
        am = []
        for i in range(self.wfn.basisset().nshell()):
            shell = self.wfn.basisset().shell(i)
            am.append(shell.am)
        am = np.asarray(am)
        return am.max()

    def get_total_charge(self):
        """
        Get total charge of the system
        """
        return self.mol.molecular_charge()

    def get_nuclear_charge(self):
        """
        Returns sum of nuclear charges

        Return
        --------
        nuclear_charge: float
            Sum of the nuclear charges in the system
        """
        nuclear_charge = 0
        for i in range(self.mol.natom()):
            nuclear_charge += self.mol.charge(i)
        return nuclear_charge

    def get_Z(self):
        """
        Individual nuclear charges
        """
        nuclear_charge = []
        for i in range(self.mol.natom()):
            nuclear_charge.append(self.mol.charge(i))
        return np.asarray(nuclear_charge)

    def get_center_position(self, center_index):
        """
        Return system positions
        Parameters
        -----------
        center_index: int
            Index of atomic center that corresponds to a basis function

        Returns
        -----------
        np.array
            Array of atom positions in Bohr
        """
        return self.pos[center_index]

    def cartesian2spherical(self):
        """
        Similar to PySCF, forms a matrix for converting
        an integral matrix from Cartesian to Spherical orbitals
        """
        #1s orbital needs no conversion
        c2sph = [np.asarray([1.0])]

        #p orbitals also need no conversion, but this order
        #converts to the same order as Psi4
        p_conv = np.asarray(
                    [[0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0]]
                    )
        
        c2sph.append(p_conv)

        #d orbital conversion from cartesian to spherical
        d_conv = np.zeros((6, 5))
        d_conv[:, 0] = [-0.5, 0, 0, -0.5, 0, 1.0]
        d_conv[:, 1] = [0, 0, np.sqrt(3), 0, 0, 0]
        d_conv[:, 2] = [0, 0, 0, 0, np.sqrt(3), 0]
        d_conv[:, 3] = [np.sqrt(3)/2, 0, 0, -np.sqrt(3)/2, 0, 0]
        d_conv[:, 4] = [0, np.sqrt(3), 0, 0, 0, 0]
        c2sph.append(d_conv)

        #Loops through the shells and assembles the full matrix
        #for converting from cartesian to spherical
        coeff_matrix = []
        for i in range(self.wfn.basisset().nshell()):
            shell = self.wfn.basisset().shell(i)
            am = shell.am
            coeff_matrix.append(c2sph[am])
        
        coeff_matrix = scipy.linalg.block_diag(*coeff_matrix)
        return coeff_matrix

    def reorder_int(self, ints, am_vals):
        """
        reorder integral based on ordering in am_vals

        Parameters
        -----------
        ints: np.ndarray
            Set of integrals
        am_vals: np.ndarray
            angular momentum values for accessing self.reorder dict
        
        Returns
        -----------
        ints: np.ndarray
            Reordered integral dictionary
        """
        am_vals = tuple(am_vals)
        if am_vals in list(self.reorder.keys()):
            reorder = self.reorder[am_vals]
            ints = ints[reorder]
        return ints
