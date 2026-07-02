import scipy
import scipy.sparse
import scipy.sparse.linalg as spla
from dataclasses import dataclass
import numpy as np

from collections.abc import Sequence

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






    

