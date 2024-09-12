#!/usr/bin/env python 
import numpy as np
from ase.io import read
from basis import Basis
from overlap import Overlap
from kinetic import Kinetic
from potential import Potential
from eri import ERI
from nuclear_repulsion import NuclearRepulsion
from rhf import RHF
from cphf import CPHF
from scf_functions import *
import sys
import argparse
np.set_printoptions(precision=5, linewidth=200, suppress=True)

def main():

    parser = argparse.ArgumentParser(description="""Simple RHF calculation program, can compute up to RHF Gradients so far""")
    parser.add_argument("atoms", type=str, help="Input xyz file")
    parser.add_argument("--spherical", default=False, action=argparse.BooleanOptionalAction, help="Output cartesian or spherical orbitals")

    args = parser.parse_args()
    #Load ASE atoms object
    atoms = read(args.atoms)
    #Use cartesian or spherical orbitals
    spherical = args.spherical
    
    #Define elements
    elems = atoms.get_chemical_symbols()

    #Handle basis set information
    basis = Basis(atoms, '3-21g', spherical)

    #Determines the number of electrons in the molecule
    total_charge = basis.get_total_charge()
    nuclear_charge_sum = basis.get_nuclear_charge()
    #N is the number of electrons
    N = nuclear_charge_sum - total_charge

    overlap = Overlap(basis)
    
    kinetic = Kinetic(basis)

    potential = Potential(basis)

    eri = ERI(basis)

    nuclear_repulsion = NuclearRepulsion(basis)

    rhf = RHF(overlap, kinetic, potential, eri, nuclear_repulsion, basis)
    energy = rhf.calculate_energy()

    gradient = rhf.calculate_gradient()

    print(f"Final energy = {energy} Eh")

    cphf = CPHF(overlap, kinetic, potential, eri, nuclear_repulsion, rhf, basis)
    C1 = cphf.calculate()

if __name__ == "__main__":
    main()
