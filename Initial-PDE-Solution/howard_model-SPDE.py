# -----------------------------------------------------------------------------
# Adapted from example from the py-pde official documentation:
# https://py-pde.readthedocs.io/en/latest/examples_gallery/advanced_pdes/pde_brusselator_class.html
# -----------------------------------------------------------------------------

import numpy as np

from pde import FieldCollection, PDEBase, plot_kymographs, ScalarField, CartesianGrid, FileStorage, PlotTracker, MovieStorage, movie
from tempfile import NamedTemporaryFile
import numba


class HowardPDE(PDEBase):
    use_noise_coefficient = True
    is_sde = True

    def __init__(self, diffusivity, reaction_params, nu, bc="auto_periodic_neumann"):
        super().__init__()
        self.diffusivity = diffusivity
        self.reaction_params = reaction_params
        self.nu = nu
        self.bc = bc

    def get_initial_state(self, grid):
        """Prepare a useful initial state."""
        c_d = ScalarField(grid, 1400, label="Field $c_D$")
        c_e = ScalarField(grid, 85, label="Field $c_E$")
        m_d = ScalarField(grid, 0, label="Field $m_D$")
        m_e = ScalarField(grid, 0, label="Field $m_E$") 
               
        return FieldCollection([c_d, c_e, m_d, m_e])

    def evolution_rate(self, state, t=0):
        """Pure python implementation of the PDE."""
        c_d, c_e, m_d, m_e = state
        r_da, r_dd, r_ea, r_ed, u_d1, u_e1 = self.reaction_params
        dif_d, dif_e = self.diffusivity

        rate_d_on = r_da * c_d / (1 + u_d1 * m_e)
        rate_d_off = r_dd * m_e * m_d
        rate_e_on = r_ea * c_d * c_e
        rate_e_off = r_ed * m_e / (1 + u_e1 * c_d)

        rhs = state.copy()
        rhs[0] = dif_d * c_d.laplace(self.bc) + rate_d_off - rate_d_on
        rhs[1] = dif_e * c_e.laplace(self.bc) + rate_e_off - rate_e_on
        rhs[2] = -rate_d_off + rate_d_on
        rhs[3] = -rate_e_off + rate_e_on

        return rhs

    def make_evolution_rate(self, state, backend):
        """Compilable implementation of the PDE."""
        r_da, r_dd, r_ea, r_ed, u_d1, u_e1 = self.reaction_params
        dif_d, dif_e = self.diffusivity
        laplace = state.grid.make_operator(
            "laplace", bc=self.bc, backend=backend
        )

        @numba.njit
        def pde_rhs(state_data, t):
            c_d = state_data[0]
            c_e = state_data[1]
            m_d = state_data[2]
            m_e = state_data[3]

            rate_d_on = r_da * c_d / (1 + u_d1 * m_e)
            rate_d_off = r_dd * m_e * m_d
            rate_e_on = r_ea * c_d * c_e
            rate_e_off = r_ed * m_e / (1 + u_e1 * c_d)

            rate_c_d = dif_d * laplace(c_d) + rate_d_off - rate_d_on
            rate_c_e = dif_e * laplace(c_e) + rate_e_off - rate_e_on
            rate_m_d = -rate_d_off + rate_d_on
            rate_m_e = -rate_e_off + rate_e_on

            return np.stack((rate_c_d, rate_c_e, rate_m_d, rate_m_e))

        return pde_rhs
    
    def _make_pde_rhs_numba(self, state):
        return self.make_evolution_rate(state, backend="numba")
    
    def make_noise_coefficient(self, state, backend):
        r_da, r_dd, r_ea, r_ed, u_d1, u_e1 = self.reaction_params
        print("noise coeff!")
        A = np.array([
            [-1, 1, 0, 0],
            [0, 0, -1, 1],
            [1, -1, 0, 0],
            [0, 0, 1, -1],
        ])
        nu = self.nu

        @numba.njit
        def noise_coefficient(state_data, t):
            c_d = state_data[0]
            c_e = state_data[1]
            m_d = state_data[2]
            m_e = state_data[3]

            rate_d_on = r_da * c_d / (1 + u_d1 * m_e)
            rate_d_off = r_dd * m_e * m_d
            rate_e_on = r_ea * c_d * c_e
            rate_e_off = r_ed * m_e / (1 + u_e1 * c_d)
            
            return nu * A @ np.diag(np.array([np.sqrt(rate_d_on), np.sqrt(rate_d_off), np.sqrt(rate_e_on), np.sqrt(rate_e_off)]))

        return noise_coefficient, 4
            


if __name__ == '__main__':
    # initialize state
    grid = CartesianGrid([(0, 2), (0, 2)], [64, 64], [False, False])
    eq = HowardPDE(diffusivity=[0.28, 0.6], reaction_params=[20, 6.3e-3, 4e-2, 0.8, 2.8e-2, 2.7e-2], nu=1000)
    state = eq.get_initial_state(grid)

    # run a simulation
    tracker = PlotTracker(interrupts=10)
    # initialize empty storages
    file_write = FileStorage("./temp/howard-SPDE.hdf")
    #movie_write = MovieStorage("howard-2D.avi", vmin=0, vmax=[120, 20, 1500, 100], bits_per_channel=16)

    eq.solve(state, 
        t_range=500, 
        dt=1e-4,
        solver="euler",
        backend="numba",
        tracker=["progress", tracker, file_write.tracker(1)])

    file_read = FileStorage("./temp/howard-SPDE.hdf")
    movie(file_read, filename="../Simulation-results/howard-SPDE-nu100.mov", show_time=True, movie_args={"framerate": 5})