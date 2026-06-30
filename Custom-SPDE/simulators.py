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
from grid import Grid

from typing import Callable, Any
from collections.abc import Generator, Sequence

def run_simulation_semi_implicit(dt: float, 
                   t_range: tuple[float, float], 
                   initial_state: np.ndarray, 
                   diffusion_matrix: scipy.sparse.coo_matrix, 
                   reaction_fn: Callable[[np.typing.NDArray[np.float32], float], np.typing.NDArray[np.float32]],
                   copy_every: int,
                   grid: Grid):

    I = scipy.sparse.eye(diffusion_matrix.shape[0])
    lhs = I - dt * diffusion_matrix
    solver = spla.factorized(lhs.tocsc())

    # 1. Turn on Interactive Mode
    v_grid = initial_state.reshape(grid.channels, grid.get_n())
    m_d = v_grid[2, :]
    m_d_grid = m_d.reshape(grid.shape)
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 5))

    img = ax.imshow(m_d_grid, origin='lower', cmap='viridis')
    plt.colorbar(img, ax=ax, label="Concentration")
    ax.set_title("Channel $m_d$")
    plt.show()

    steps = (t_range[1]-t_range[0])/dt

    history = [initial_state.copy()]

    cur = initial_state
    stop = True
    for k in range(int(steps)):
        reactions = reaction_fn(cur, dt)
        rhs = cur + reactions
        cur: np.typing.NDArray[np.float32] = solver(rhs)
        cur = cur.clip(0.0)

        if k % copy_every == 0:
            #history.append(cur.copy())
            print(k*dt)
            v_grid = cur.reshape(grid.channels, grid.get_n())
            m_d = v_grid[2, :]
            m_d_grid = m_d.reshape(grid.shape)
            print(m_d_grid[0,0])

            img.set_data(m_d_grid)
            
            vmin = m_d_grid.min()
            vmax = m_d_grid.max()
            vmin = vmin if vmin>0.9*vmax else 0
            vmin = vmin if vmin<0.9*vmax else 0.9*vmax
            img.set_clim(vmin=vmin, vmax=m_d_grid.max())
            
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        
        if stop and (500 <= k*dt <= 600):
            input("continue?")
            stop = False
    return history    



def run_simulation_explicit(dt: float, 
                   t_range: tuple[float, float], 
                   initial_state: np.ndarray, 
                   diffusion_matrix: scipy.sparse.csr_matrix, 
                   reaction_fn: Callable[[np.typing.NDArray[np.float32], float], np.typing.NDArray[np.float32]],
                   copy_every: int,
                   grid: Grid):

    v_grid = initial_state.reshape(grid.channels, grid.get_n())
    m_d = v_grid[2, :]
    m_d_grid = m_d.reshape(grid.shape)
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(m_d_grid, origin='lower', cmap='viridis')
    plt.colorbar(img, ax=ax, label="Concentration")
    ax.set_title("Channel $m_d$")

    steps = (t_range[1]-t_range[0])/dt

    history = [initial_state.copy()]

    cur = initial_state
    stop = True
    for k in range(int(steps)):
        reactions = reaction_fn(cur, dt)
        cur = cur + reactions + diffusion_matrix @ cur * dt
        cur = cur.clip(0.0)

        if k % copy_every == 0:
            #history.append(cur.copy())
            print(k*dt)
            v_grid = cur.reshape(grid.channels, grid.get_n())
            m_d = v_grid[2, :]
            m_d_grid = m_d.reshape(grid.shape)
            img.set_data(m_d_grid)
            
            vmin = m_d_grid.min()
            vmax = m_d_grid.max()
            vmin = vmin if vmin>0.9*vmax else 0
            vmin = vmin if vmin<0.9*vmax else 0.9*vmax
            img.set_clim(vmin=vmin, vmax=m_d_grid.max())
            
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        
        if stop and (500 <= k*dt <= 600):
            input("continue?")
            stop = False
    return history    