
import numpy as np
from ase import Atoms
from typing import List, Tuple, Union

class CVLogger:
    """
    Logs the state of the collective variable and other parameters in Metadynamics.

    Parameters
    ----------
    hills_file:
        Name of the output hills log file.

    log_period:
        Time steps between logging of collective variables and Metadynamics parameters.
    """

    def __init__(self, log_file, log_period):
        """
        MetaDLogger constructor.
        """
        self.log_file = log_file
        self.log_period = log_period
        self.counter = 0

    def save_cvs(self, cv):
        """
        Store CV values
        """
        with open(f'{self.log_file}.dat', "a+", encoding="utf8") as f:
            f.write(str(self.counter) + "\t")
            f.write("\t".join(map(str, cv.flatten())) + "\n")

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        if self.counter >= self.log_period and self.counter % self.log_period == 0:
            self.save_cvs(state.xi)

        self.counter += 1



def dihedrals(traj: List[Atoms], indices_list: list[Tuple]):
    """
    Indices should be a list of tuple that has 4 components
    """
    result_list = []
    for atoms in traj:
        result = []
        for indices in indices_list:
            i, j, k, l = indices
            pos_i = atoms.positions[i]
            pos_j = atoms.positions[j]
            pos_k = atoms.positions[k]
            pos_l = atoms.positions[l]

            vec_ij = pos_j - pos_i  # i -> j
            vec_jk = pos_k - pos_j  # j -> k
            vec_kl = pos_l - pos_k  # k -> l

            normal1 = np.cross(vec_ij, vec_jk)
            normal2 = np.cross(vec_jk, vec_kl)

            normal1_norm = normal1 / np.linalg.norm(normal1)
            normal2_norm = normal2 / np.linalg.norm(normal2)

            value = np.dot(normal1_norm, normal2_norm)
            cross_prod = np.cross(normal1_norm, normal2_norm)
            sign = np.dot(cross_prod, vec_jk / np.linalg.norm(vec_jk))

            value = np.clip(value, -1.0, 1.0)
            angle = np.arccos(value)

            if sign < 0:
                angle = -angle
            
            result.append(angle)
        result_list.append(result)
    return np.array(result_list)