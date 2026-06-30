import jax
import jax.scipy.sparse.linalg
import scipy
import scipy.sparse
import scipy.sparse.linalg as spla
from dataclasses import dataclass
import itertools as it
import numpy as np
import numpy.typing
import matplotlib.pyplot as plt

from typing import Callable, Any
from collections.abc import Generator, Sequence

#I will use semi-implicit Euler in Jax, I need to have the finite difference laplacian, which we then do a step via solving.
#At each time step we solve implicitly with euler but add the noise with 

@dataclass()
class Grid:
    bounds: Sequence[tuple[float, float]]
    shape: Sequence[int]
    channels: int

    def get_cell_dimensions(self):
        h = zip(self.bounds, self.shape)
        return [(dmax-dmin)/(n-1) for (dmin, dmax), n in h]
    
    def get_n(self):
        return int(np.prod(self.shape))
        

def create_discrete_diffusion_matrix(grid: Grid, diffusion_coeffs):
    laplacian = spla.LaplacianNd(grid.shape, boundary_conditions='neumann').tosparse()
    zeros = scipy.sparse.csr_matrix((grid.get_n(), grid.get_n()), dtype=np.float32)
    mats = [
        (coeff/(grid.get_cell_dimensions()[0]**2)) * laplacian if coeff != 0
        else zeros
        for coeff in diffusion_coeffs
    ]

    return scipy.sparse.block_diag(mats)



if __name__ == "__main__":
    pass






    

