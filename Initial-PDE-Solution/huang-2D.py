# -----------------------------------------------------------------------------
# Adapted from example from the py-pde official documentation:
# https://py-pde.readthedocs.io/en/latest/examples_gallery/advanced_pdes/pde_brusselator_class.html
# -----------------------------------------------------------------------------

import numpy as np

from pde import FieldCollection, PDEBase, plot_kymographs, ScalarField, CartesianGrid, FileStorage, PlotTracker, MovieStorage, movie
from tempfile import NamedTemporaryFile
import numba


class HuangPDE(PDEBase):
    def __init__(self, diffusivity, reaction_params, h, bc="auto_periodic_neumann"):
        super().__init__()
        self.diffusivity = diffusivity
        self.reaction_params = reaction_params
        self.h = h
        self.bc = bc

    def get_initial_state(self, grid):
        """Prepare a useful initial state."""
        c_d_adp = ScalarField(grid, 0, label="Field $c_{D-ADP}$")
        c_d_atp = (ScalarField(grid, 1300, label="Field $c_{D-ATP}$")
               + 100 * ScalarField.random_normal(grid, label="Field $c_{D-TDP}$"))
        c_e = ScalarField(grid, 115, label="Field $c_E$")
        m_d = ScalarField(grid, 0, label="Field $m_D$")
        m_de = ScalarField(grid, 0, label="Field $m_{DE}$") 
               
        return FieldCollection([c_d_adp, c_d_atp, c_e, m_d, m_de])

    def evolution_rate(self, state, t=0):
        """Pure python implementation of the PDE."""
        c_d_adp, c_d_atp, c_e, m_d, m_de = state
        k_ne, r_d_on0, r_d_on1_d, r_d_on1_de, r_e_on1_d, r_de_off = self.reaction_params
        dif_d, dif_e = self.diffusivity
        h = self.h

        r_d_on = r_d_on0 + r_d_on1_d * m_d + r_d_on1_de * m_de
        r_e_on = r_e_on1_d * m_d

        rate_ne = k_ne * c_d_adp
        rate_d_on = r_d_on * c_d_atp
        rate_e_on = r_e_on * c_e
        rate_de_off = r_de_off * m_de
        
        rhs = state.copy()
        rhs[0] = dif_d * c_d_adp.laplace(self.bc) - rate_ne + rate_de_off/h
        rhs[1] = dif_d * c_d_atp.laplace(self.bc) + rate_ne - rate_d_on/h
        rhs[2] = dif_e * c_e.laplace(self.bc) - rate_e_on/h + rate_de_off/h
        rhs[3] = rate_d_on - rate_e_on
        rhs[4] = rate_e_on - rate_de_off

        return rhs

    def make_evolution_rate(self, state, backend):
        """Compilable implementation of the PDE."""
        k_ne, r_d_on0, r_d_on1_d, r_d_on1_de, r_e_on1_d, r_de_off = self.reaction_params
        dif_d, dif_e = self.diffusivity
        h = self.h

        laplace = state.grid.make_operator(
            "laplace", bc=self.bc, backend=backend
        )

        @numba.njit
        def pde_rhs(state_data, t):
            c_d_adp = state_data[0]
            c_d_atp = state_data[1]
            c_e = state_data[2]
            m_d = state_data[3]
            m_de = state_data[4]

            r_d_on = r_d_on0 + r_d_on1_d * m_d + r_d_on1_de * m_de
            r_e_on = r_e_on1_d * m_d

            rate_ne = k_ne * c_d_adp
            rate_d_on = r_d_on * c_d_atp
            rate_e_on = r_e_on * c_e
            rate_de_off = r_de_off * m_de

            rate_c_d_adp = dif_d * laplace(c_d_adp) - rate_ne + rate_de_off/h
            rate_c_d_atp = dif_d * laplace(c_d_atp) + rate_ne - rate_d_on/h
            rate_c_e = dif_e * laplace(c_e) - rate_e_on/h + rate_de_off/h
            rate_m_d = rate_d_on - rate_e_on
            rate_m_de = rate_e_on - rate_de_off

            return np.stack((rate_c_d_adp, rate_c_d_atp, rate_c_e, rate_m_d, rate_m_de))

        return pde_rhs
    
    def _make_pde_rhs_numba(self, state):
        return self.make_evolution_rate(state, backend="numba")

if __name__ == '__main__':
    # initialize state
    grid = CartesianGrid([(0, 1), (0, 16)], [16, 256], [False, False])
    eq = HuangPDE(diffusivity=[2.5, 2.5], reaction_params=[1, 0.025, 0.0015, 0.0015, 0.093, 0.7], h=1)
    state = eq.get_initial_state(grid)

    # run a simulation
    tracker = PlotTracker(interrupts=10)
    # initialize empty storages
    file_write = FileStorage("./temp/huang-2D-16x1-5000-20260625-2357.hdf")
    #movie_write = MovieStorage("howard-2D.avi", vmin=0, vmax=[120, 20, 1500, 100], bits_per_channel=16)

    eq.solve(state, 
        t_range=2000, 
        solver="scipy",
        backend="numba",
        tracker=["progress", tracker, file_write.tracker(1)])

    file_read = FileStorage("./temp/huang-2D-16x1-5000-20260625-2357.hdf")
    movie(file_read, filename="../Simulation-results/huang-2D-16x1-5000-20260625-2357.mov", show_time=True, movie_args={"framerate": 5})