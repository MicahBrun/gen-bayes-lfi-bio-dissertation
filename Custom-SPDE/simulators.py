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
                   call_every: tuple[int, Callable[[float, np.typing.NDArray[np.float32]], None]]):

    call_interval, call = call_every

    I = scipy.sparse.eye(diffusion_matrix.shape[0])
    lhs = I - dt * diffusion_matrix
    solver = spla.factorized(lhs.tocsc())

    steps = (t_range[1]-t_range[0])/dt
    cur = initial_state
    for k in range(int(steps)):
        reactions = reaction_fn(cur, dt)
        rhs = cur + reactions
        cur: np.typing.NDArray[np.float32] = solver(rhs)
        cur = cur.clip(0.0)

        if k % call_interval == 0:
            call(k*dt, cur)



def run_simulation_explicit(dt: float, 
                   t_range: tuple[float, float], 
                   initial_state: np.ndarray, 
                   diffusion_matrix: scipy.sparse.csr_matrix, 
                   reaction_fn: Callable[[np.typing.NDArray[np.float32], float], np.typing.NDArray[np.float32]],
                   call_every: tuple[int, Callable[[float, np.typing.NDArray[np.float32]], None]]):
    call_interval, call = call_every

    steps = (t_range[1]-t_range[0])/dt
    cur = initial_state
    for k in range(int(steps)):
        reactions = reaction_fn(cur, dt)
        cur = cur + reactions + diffusion_matrix @ cur * dt
        cur = cur.clip(0.0)

        if k % call_interval == 0:
            call(k*dt, cur)