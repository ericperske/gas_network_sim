from typing import Sequence, Literal, Union, Tuple, Callable
from functools import partial
import jax
import jaxlib
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gasnetwork.topology import Node, Pipe
from gasnetwork.runge_kutta import RungeKutta, PartitionedRungeKutta
from gasnetwork.basis_functions import R_PiecewiseLinearContinuous
from gasnetwork.model import E_pip, J_pip, R_pip, B_pip, z_pip, build_node_split, ModelParams
from gasnetwork.equations import f_rk, g_rk, f_prk, g_prk, phi_prk, pressure_to_density, boundary_enthalpy

class GasNetworkSim:
    """
    This is the heart of the gas_network_sim repository. The class only receives the gas
    network itself as input in the form of nodes and pipes as well as a clarification whether 
    the network shall be solved using the Runge-Kutta implementation or the 
    partition Runge-Kutta method. Moreover the correct Butcher Tableau for the method that 
    should be used has to be declared. 
    Also, the class needs an initial value as well as a terminal time T and a step size h.

    The functions needed for the selected method are built automatically using the ._build() 
    method and will be solved using the ._simulate() method.
    """
    def __init__(self, nodes: Sequence[Node], pipes: Sequence[Pipe], method: Literal["RungeKutta", "PartitionedRungeKutta"],
                 A: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], b: jnp.ndarray, c: jnp.ndarray,
                 t0: float, T: float, h: float, q1: Callable, h2: Callable, x0=None, ALPHA=1e-5):

        self.method = method
        # Quick sanity check if Butcher Tableau corresponds to the selected method, at least dimensionally
        if self.method == "PartitionedRungeKutta":
            if not (isinstance(A, tuple) and len(A) == 2 and all(isinstance(a, jnp.ndarray) for a in A)):
                raise TypeError(f"Expected a tuple of two jnp.ndarrays for method {self.method}, got {type(A)!r}")
        elif self.method == "RungeKutta":
            if not isinstance(A, jnp.ndarray):
                raise TypeError(f"Expected a jnp.ndarray for method {self.method}, got {type(A)!r}")
        
        self.nodes = nodes
        self.pipes = pipes

        self.A = A
        self.b = b
        self.c = c
        self.ALPHA = ALPHA
        self.t0 = t0
        self.T = T
        self.h = h

        # basis function
        base_r = R_PiecewiseLinearContinuous(pipes=pipes)
        
        # node splitting
        split = build_node_split(nodes)
        self.q1_shape = split.q_1.shape
        # Define B and split into B_1 and B_2
        B = B_pip(nodes, pipes, basis_function=base_r)
        B_1, B_2 = B.split(split)

        # Define E, R, z as callables in c
        E_fun = E_pip(nodes, pipes, basis_function=base_r).build
        R_fun = R_pip(nodes, pipes, basis_function=base_r).build
        z_fun = z_pip(nodes, pipes, alpha=ALPHA).build
        
        # Define J
        J = J_pip(nodes, pipes).build()

        self.params=ModelParams(
            J=J,
            E=E_fun,
            R=R_fun,
            z_func=z_fun,
            B_1=B_1,
            B_2=B_2,
            q_1=q1,
            h_2=h2
        )

        if method == "RungeKutta":
            self.f = partial(f_rk, p=self.params)
            self.g = partial(g_rk, p=self.params)
            self.phi = None

        elif method == "PartitionedRungeKutta":
            self.f = partial(f_prk, p=self.params)
            self.g = partial(g_prk, p=self.params)
            self.phi = partial(phi_prk, p=self.params)

        
    def _build(self, SUPPLY_BAR: float, tol: float, maxiter: int):
        given_enthalpy = next(node.enthalpy for node in self.nodes if node.is_enthalpy_node)
        assert given_enthalpy is not None

        if self.method == "RungeKutta":
            x_rho0 = jnp.concatenate([
                jnp.full(pipe.n_elements, pressure_to_density(SUPPLY_BAR, self.ALPHA))
                for pipe in self.pipes
            ])
            x_q0 = jnp.array([
                val
                for pipe in self.pipes
                for val in (
                    [pipe.innode.flux if pipe.innode.is_flux_node else 0.0]
                    + [0.0] * (pipe.n_fluxes - 2)
                    + [pipe.outnode.flux if pipe.outnode.is_flux_node else 0.0]
                )
            ])
            y0 = jnp.concatenate([x_rho0, x_q0])

            # project y0 onto the algebraic constraint manifold g(y0, t0) = 0: the
            # physically-constructed y0 above only approximately satisfies it
            # (residual approx. 1e-6/1e-7, from uniform density + boundary-only fluxes not
            # being an exact solution). Mirrors the same projection done for
            # PartitionedRungeKutta's v0 (see the PRK branch below), so both methods
            # start from a mutually consistent state. Correct
            # the q-block of y0 along B_1's q-block image (the same direction the
            # multiplier pushes q through in the dynamics), solving the resulting
            # n_lambda-sized system for delta. g is not affine in y (the kinetic
            # term in z_func is quadratic in q), so this is a short Newton iteration
            # rather than a single linear solve.
            n_rho = x_rho0.shape[0]
            B_1_q = self.params.B_1[n_rho:, :]

            def constraint_of_delta(delta_):
                y_ = jnp.concatenate([y0[:n_rho], y0[n_rho:] + B_1_q @ delta_])
                return self.g(y_, self.t0)

            delta = jnp.zeros(self.q1_shape)
            for _ in range(maxiter):
                r = constraint_of_delta(delta)
                if jnp.max(jnp.abs(r)) <= tol:
                    break
                J_delta = jax.jacfwd(constraint_of_delta)(delta)
                delta = delta - jnp.linalg.solve(J_delta, r)
            y0 = jnp.concatenate([y0[:n_rho], y0[n_rho:] + B_1_q @ delta])

            # consistent w0: solve d/dt[g(y0, t0)] = 0 for w0, i.e. the hidden
            # constraint obtained by differentiating the algebraic constraint once
            # (g is affine in w here, so this is a single linear solve).
            def hidden_constraint(w_): # type: ignore
                y_dot = self.f(y0, w_, self.t0)
                _, dg = jax.jvp(lambda yy, tt: self.g(yy, tt), (y0, self.t0), (y_dot, 1.0))
                return dg

            w_guess = jnp.zeros(self.q1_shape)
            J_w = jax.jacfwd(hidden_constraint)(w_guess)
            w0 = jnp.linalg.solve(J_w, -hidden_constraint(w_guess))

            x0 = (y0, w0)
            rk = RungeKutta(A=jnp.array(self.A), b=self.b, c=self.c, f=self.f, g=self.g, newton_tol=tol, newton_maxiter=maxiter) # type checker fails if A is not specified
        elif self.method == "PartitionedRungeKutta":
            x_rho0 = jnp.concatenate([
                jnp.full(pipe.n_elements, pressure_to_density(SUPPLY_BAR, self.ALPHA))
                for pipe in self.pipes
            ])
            x_q0 = jnp.array([
                val
                for pipe in self.pipes
                for val in (
                    [pipe.innode.flux if pipe.innode.is_flux_node else 0.0]
                    + [0.0] * (pipe.n_fluxes - 2)
                    + [pipe.outnode.flux if pipe.outnode.is_flux_node else 0.0]
                )
            ])
            
            u0 = x_rho0
            v0 = x_q0

            # project v0 onto the algebraic constraint manifold phi(u0, v0, t0) = 0.
            # the physically-constructed v0 above only approximately satisfies it
            # (residual ~1e-6/1e-7, from uniform density + boundary-only fluxes not
            # being an exact solution), which otherwise leaves a brief transient at
            # t0 until the first RK step's own Newton solve absorbs it. Correct v0
            # along B_1's q-block image solving the resulting n_lambda-sized system for delta.
            # phi is not affine in v (the kinetic term in z_func is quadratic in v),
            # so this is a short Newton iteration rather than a single linear solve.
            n_rho = u0.shape[0]
            B_1_q = self.params.B_1[n_rho:, :]

            def constraint_of_delta(delta_):
                return self.phi(u0, v0 + B_1_q @ delta_, self.t0) # type: ignore

            delta = jnp.zeros(self.q1_shape)
            for _ in range(maxiter):
                r = constraint_of_delta(delta)
                if jnp.max(jnp.abs(r)) <= tol:
                    break
                J_delta = jax.jacfwd(constraint_of_delta)(delta)
                delta = delta - jnp.linalg.solve(J_delta, r)
            v0 = v0 + B_1_q @ delta

            # consistent lam0: solve d/dt[phi(u0, v0, t0)] = 0 for lam0, i.e. the
            # hidden constraint obtained by differentiating the algebraic constraint
            # once (phi is affine in lam here, so this is a single linear solve).
            def hidden_constraint(lam_):
                u_dot = self.f(u0, v0, self.t0)
                v_dot = self.g(u0, v0, lam_, self.t0)# type: ignore
                _, dphi = jax.jvp(lambda uu, vv, tt: self.phi(uu, vv, tt), (u0, v0, self.t0), (u_dot, v_dot, 1.0)) # type: ignore
                return dphi

            lam_guess = jnp.zeros(self.q1_shape)
            J_lam = jax.jacfwd(hidden_constraint)(lam_guess)
            lam0 = jnp.linalg.solve(J_lam, -hidden_constraint(lam_guess))

            x0 = (u0, v0, lam0)
            rk = PartitionedRungeKutta(As=list(self.A), b=self.b, c=self.c, f=self.f, g=self.g, phi=self.phi, newton_tol=tol, newton_maxiter=maxiter) # again: type checker fails if A is not specified further

        return x0, rk


    def simulate(self, SUPPLY_BAR: float, tol: float = 1e-6, maxiter: int = 1000):
        """
        The Runge-Kutta method for solving the gas network equations.
        """

        x0, rk = self._build(SUPPLY_BAR=SUPPLY_BAR, tol=tol, maxiter=maxiter)

        if isinstance(rk, RungeKutta):
            ts, ys, ws = rk.solve(y0=x0[0], w0=x0[1], t0=self.t0, T=self.T, h=self.h)
            return ts, ys, ws
        elif isinstance(rk, PartitionedRungeKutta) and len(x0) == 3:
            ts, us, vs, lams = rk.solve(u0=x0[0], v0=x0[1], lambda0=x0[2], t0=self.t0, T=self.T, h=self.h)
            return ts, us, vs, lams
        else:
            raise ValueError(f"Either rk is of a wrong instance or len(x0) != 3 (got len(x0) = {len(x0)})")
