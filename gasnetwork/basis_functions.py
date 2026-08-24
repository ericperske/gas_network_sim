from typing import Sequence
import numpy as np

from gasnetwork.topology import  Pipe

class P_PiecewiseConstant:
    """
    A piecewise constant basis function defined on the pipes. The function is one on exactly one pipe and zero elsewhere.
    The function is used as the discretization space P_h in the thesis.
    """

    def __init__(self, pipes: Sequence[Pipe]):
        self.pipes = pipes

    def evaluate(self, x: np.ndarray, j: int) -> float:
        return 1.0 if self.pipes[j-1] <= x < self.pipes[j] else 0.0
    

class R_PiecewiseLinearContinuous:
    """
    A piecewise linear continuous basis function defined locally on each pipe's own
    sub-element mesh (see Pipe.flux_refinement). local_idx runs over the pipe's local
    node numbering 0,...,pipe.n_fluxes-1; local_idx 0 and pipe.n_elements are the pipe's
    real innode/outnode, everything in between is an interior coefficient private to the pipe.
    """

    def __init__(self, pipes: Sequence[Pipe]):
        self.pipes = pipes

    def evaluate(self, x: float, local_idx: int, pipe_idx: int) -> float:
        pipe = self.pipes[pipe_idx]
        h = pipe.element_length
        xk = local_idx * h

        if local_idx > 0 and xk - h <= x <= xk:
            return (x - (xk - h)) / h
        elif local_idx < pipe.n_elements and xk <= x <= xk + h:
            return (xk + h - x) / h
        else:
            return 0.0

    def derivative(self, x: float, local_idx: int, pipe_idx: int) -> float:
        pipe = self.pipes[pipe_idx]
        h = pipe.element_length
        xk = local_idx * h

        if local_idx > 0 and xk - h <= x <= xk:
            return 1 / h
        elif local_idx < pipe.n_elements and xk <= x <= xk + h:
            return -1 / h
        else:
            return 0.0
