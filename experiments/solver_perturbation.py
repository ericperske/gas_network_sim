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

class GasNetworkSimPerturbation:
    """
    Variant of gasnetwork.solver.GasNetworkSim for the perturbation-of-initial-data
    experiments. Everything is identical to GasNetworkSim except _build(), which takes
    an extra theta_0 argument: after Newton-projecting y0 (RungeKutta) / v0
    (PartitionedRungeKutta) onto the algebraic constraint manifold g(y0,t0)=0 /
    phi(u0,v0,t0)=0 exactly like GasNetworkSim, it then pushes the corrected value back
    off the manifold along the same direction, sized (via the constraint's own local
    Jacobian) so that the resulting defect (either ||g(y0,t0)||_2 or ||phi(u0,v0,t0)||_2)
    is theta_0. theta_0 = 0.0 reproduces GasNetworkSim's fully consistent initial data.

    Since g/phi is only approximately affine in the perturbed direction (the kinetic
    term in z_func is quadratic), the resulting defect matches theta_0 to first order.

    w0 / lambda0 are still obtained by solving the hidden constraint (differentiating
    g / phi once) at the (possibly perturbed) y0/v0, exactly as in GasNetworkSim.
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


    def _build(self, SUPPLY_BAR: float, tol: float, maxiter: int, theta_0: float = 0.0):
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

            # project y0 onto the algebraic constraint manifold g(y0, t0) = 0, exactly
            # as GasNetworkSim does (see its _build() for the full rationale). Correct
            # the q-block of y0 along B_1's q-block image, solving the resulting
            # n_lambda-sized system for delta via a short Newton iteration (g is not
            # affine in y).
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

            # theta_0 perturbation: starting from the now-consistent delta, move a
            # further step along a fixed unit direction, sized via the constraint's
            # local Jacobian (a linear map from delta-space to defect-space) so that
            # the resulting defect g(y0, t0) has L2 norm theta_0.
            if theta_0 != 0.0:
                direction = jnp.ones(self.q1_shape)
                direction = direction / jnp.linalg.norm(direction)
                J_delta = jax.jacfwd(constraint_of_delta)(delta)
                delta = delta + jnp.linalg.solve(J_delta, theta_0 * direction)

            y0 = jnp.concatenate([y0[:n_rho], y0[n_rho:] + B_1_q @ delta])

            # consistent w0: solve d/dt[g(y0, t0)] = 0 for w0, i.e. the hidden
            # constraint obtained by differentiating the algebraic constraint once
            # (g is affine in w here, so this is a single linear solve). Independent
            # of whether y0 itself sits exactly on the manifold.
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

            # project v0 onto the algebraic constraint manifold phi(u0, v0, t0) = 0,
            # exactly as GasNetworkSim does (see its _build() for the full rationale).
            # Correct v0 along B_1's q-block image, solving the resulting
            # n_lambda-sized system for delta via a short Newton iteration (phi is not
            # affine in v).
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

            # theta_0 perturbation: starting from the now-consistent delta, move a
            # further step along a fixed unit direction, sized via the constraint's
            # local Jacobian (a linear map from delta-space to defect-space) so that
            # the resulting defect phi(u0, v0, t0) has L2 norm theta_0.
            if theta_0 != 0.0:
                direction = jnp.ones(self.q1_shape)
                direction = direction / jnp.linalg.norm(direction)
                J_delta = jax.jacfwd(constraint_of_delta)(delta)
                delta = delta + jnp.linalg.solve(J_delta, theta_0 * direction)

            v0 = v0 + B_1_q @ delta

            # consistent lam0: solve d/dt[phi(u0, v0, t0)] = 0 for lam0, i.e. the
            # hidden constraint obtained by differentiating the algebraic constraint
            # once (phi is affine in lam here, so this is a single linear solve).
            # Independent of whether v0 itself sits exactly on the manifold.
            def hidden_constraint(lam_):
                u_dot = self.f(u0, v0, self.t0)
                v_dot = self.g(u0, v0, lam_, self.t0) # type: ignore
                _, dphi = jax.jvp(lambda uu, vv, tt: self.phi(uu, vv, tt), (u0, v0, self.t0), (u_dot, v_dot, 1.0)) # type: ignore
                return dphi

            lam_guess = jnp.zeros(self.q1_shape)
            J_lam = jax.jacfwd(hidden_constraint)(lam_guess)
            lam0 = jnp.linalg.solve(J_lam, -hidden_constraint(lam_guess))

            x0 = (u0, v0, lam0)
            rk = PartitionedRungeKutta(As=list(self.A), b=self.b, c=self.c, f=self.f, g=self.g, phi=self.phi, newton_tol=tol, newton_maxiter=maxiter) # again: type checker fails if A is not specified further

        return x0, rk


    def simulate(self, SUPPLY_BAR: float, tol: float = 1e-6, maxiter: int = 1000, theta_0: float = 0.0):
        """
        The Runge-Kutta method for solving the gas network equations.

        theta_0 perturbs the initial data off the algebraic constraint manifold: the
        L2 norm of g(y0,t0) (RungeKutta) / phi(u0,v0,t0) (PartitionedRungeKutta) is
        driven to (approximately) theta_0 instead of 0. Defaults to 0.0, i.e. fully
        consistent initial data.
        """

        x0, rk = self._build(SUPPLY_BAR=SUPPLY_BAR, tol=tol, maxiter=maxiter, theta_0=theta_0)

        if isinstance(rk, RungeKutta):
            ts, ys, ws = rk.solve(y0=x0[0], w0=x0[1], t0=self.t0, T=self.T, h=self.h)
            return ts, ys, ws
        elif isinstance(rk, PartitionedRungeKutta) and len(x0) == 3:
            ts, us, vs, lams = rk.solve(u0=x0[0], v0=x0[1], lambda0=x0[2], t0=self.t0, T=self.T, h=self.h)
            return ts, us, vs, lams
        else:
            raise ValueError(f"Either rk is of a wrong instance or len(x0) != 3 (got len(x0) = {len(x0)})")
