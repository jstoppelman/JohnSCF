import numpy as np

class NuclearRepulsion:
    def __init__(self, basis):
        self.basis = basis

        self.positions = self.basis.pos
        self.Z = self.basis.get_Z()

    def calculate_nuclear_repulsion(self):
        nuclear_repulsion = 0.0
        for i in range(self.positions.shape[0]):
            for j in range(i+1, self.positions.shape[0]):
                R = (((self.positions[i] - self.positions[j])**2).sum(axis=0))**(0.5)
                nuclear_repulsion += self.Z[i] * self.Z[j] / R
        return nuclear_repulsion

    def calculate_gradient(self, atom, component):
        gradient = 0
        for i in range(self.positions.shape[0]):
            if i != atom:
                R = (((self.positions[i] - self.positions[atom])**2).sum(axis=0))**(0.5)
                gradient += (self.positions[i, component] - self.positions[atom, component]) * self.Z[i] * self.Z[atom]/R**3
        
        return gradient

