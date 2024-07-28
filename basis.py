import numpy as np
import psi4

class Basis:
    """
    Use Psi4 to read in basis set info 
    and store it in class
    """
    def __init__(self, atoms, basis):
        """
        Parameters
        -----------
        atoms: Atoms object
            contains system info
        """
        self.atoms = atoms

        #Should only need to input basis info here
        psi4.set_options({'basis': basis})

        #Form Psi4 mol object
        symbols = self.atoms.get_chemical_symbols()
        pos = self.atoms.get_positions()
        mol_str = ''
        for s, xyz in zip(symbols, pos):
            mol_str += f'{s}  {xyz[0]} {xyz[1]} {xyz[2]}\n'
        mol_str += 'units angstrom\n'
        self.mol = psi4.geometry(mol_str)
        
        #Make stored wavefunction
        self.wfn = psi4.core.Wavefunction.build(self.mol, psi4.core.get_global_option('BASIS'))

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
