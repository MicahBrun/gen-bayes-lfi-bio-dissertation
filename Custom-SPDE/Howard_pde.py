from grid import Grid, create_discrete_diffusion_matrix
from simulators import run_simulation_semi_implicit, run_simulation_explicit
import numpy as np

def make_dW(grid: Grid):
    dV = np.prod(grid.get_cell_dimensions())
    def dW(dt):
        return np.random.standard_normal(grid.get_n()).astype(np.float32)*(np.sqrt(dt)*np.sqrt(dV))
    return dW

def get_dreaction(rate, dW, dt, sigma):
    return rate*dt + sigma*np.sqrt(rate)*dW(dt)

def make_reactions_fn(reaction_params: tuple[float, float, float, float, float, float, ], sigma: float, grid: Grid):
    r_da, r_dd, r_ea, r_ed, u_d1, u_e1 = reaction_params

    dW = make_dW(grid)

    out = np.empty((grid.channels, grid.get_n()), dtype=np.float32)
    def fn(state: np.typing.NDArray[np.float32], dt):
        state_channels = state.reshape(grid.channels, grid.get_n())
        c_d = state_channels[0, :]
        c_e = state_channels[1, :]
        m_d = state_channels[2, :]
        m_e = state_channels[3, :]
        
        rate_d_on = r_da * c_d / (1 + u_d1 * m_e)
        rate_d_off = r_dd * m_e * m_d
        rate_e_on = r_ea * c_d * c_e
        rate_e_off = r_ed * m_e / (1 + u_e1 * c_d)

        d_d_on = get_dreaction(rate_d_on, dW, dt, sigma)
        d_d_off = get_dreaction(rate_d_off, dW, dt, sigma)
        d_e_on = get_dreaction(rate_e_on, dW, dt, sigma)
        d_e_off = get_dreaction(rate_e_off, dW, dt, sigma)

        out[0, :] = d_d_off - d_d_on
        out[1, :] = d_e_off - d_e_on
        out[2, :] = -d_d_off + d_d_on
        out[3, :] = -d_e_off + d_e_on

        return out.reshape(-1)

    return fn

if __name__ == '__main__':
    grid = Grid([(0,2), (0,1)], [64+1, 32+1], 4)
    diffusion = create_discrete_diffusion_matrix(grid, [0.28, 0.6, 0, 0])
    reactions_fn = make_reactions_fn((20, 6.3e-3, 4e-2, 0.8, 2.8e-2, 2.7e-2), 10, grid)

    initial_state = np.zeros((grid.channels, grid.get_n()), dtype=np.float32)
    noise = np.random.standard_normal(grid.get_n()).astype(np.float32)
    initial_state[0, :] = 1400.0 + 100.0 * noise
    initial_state[1, :] = 85.0

    initial_state = initial_state.reshape(-1)
    dt = 1e-3
    results = run_simulation_semi_implicit(dt, (0, 10000), initial_state, diffusion.tocsr(), reactions_fn, int(1/dt), grid)