# JohnSCF

Simple SCF program written all in Python. Can currently compute the energy and gradient. The integrals are all computed using the Obara-Saika scheme (see J. Chem. Phys. 1986, 84, 3963) In principle, this code will work with basis functions possessing up to d-level angular momentum. However, small modification to the basis function ordering code in basis.py should allow for arbitrary angular momentum calculations.
