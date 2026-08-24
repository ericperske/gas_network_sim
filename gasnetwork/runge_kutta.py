import jax.numpy as jnp
import jax
from functools import partial
from typing import Sequence

class RungeKutta:
    """
    Generic implicit Runge-Kutta method for solving the Hessenberg-System

        y' = f(y,w,t),
        0 = g(y,t),

    for a given Butcher tableau (A,b,c) with len(b) = s stages.
    This class supports an arbitrary amount of stages but is indented for using implicit tableaus only
    that are also stiffly-accurate.
    The steps are solved using a newton method with a given tolerance and a maximum number of iterations.

    """
    def __init__(self, A: jax.Array, b: jax.Array, c: jax.Array, f, g, newton_tol: float = 1e-6, newton_maxiter: int = 40):
        self.A = A
        self.b = b
        self.c = c
        self.s = len(b)
        self.f = f
        self.g = g
        self.newton_tol = newton_tol
        self.newton_maxiter = newton_maxiter

    def _residual(self, YW, eta, t0, h, ny, nw):
        Y = YW[: self.s * ny].reshape(self.s, ny)
        W = YW[self.s * ny :].reshape(self.s, nw)
        t_stages = t0 + self.c * h

        f_vmap = jax.vmap(self.f, in_axes=(0, 0, 0))
        g_vmap = jax.vmap(self.g, in_axes=(0,0))

        F = f_vmap(Y, W, t_stages)
        R_Y = Y - eta[None, :] - h * (self.A @ F)
        R_g = g_vmap(Y, t_stages)

        return jnp.concatenate([R_Y.reshape(-1), R_g.reshape(-1)])


    def step(self, y0: jax.Array, w0: jax.Array, t0: float, h: float):
        """
        Performs a single Runge-Kutta step from t0 to t0 + h with initial values y0 and w0.

        Returns:
            y1 : array, shape (ny, )
            w1 : array, shape (nw, )
        """
        ny = y0.shape[0]
        nw = w0.shape[0]

        Y0 = jnp.tile(y0, (self.s, 1)) # Initial guess for Y stages
        W0 = jnp.tile(w0, (self.s, 1)) # Initial guess for W stages
        x0 = jnp.concatenate([Y0.reshape(-1), W0.reshape(-1)])

        residual = partial(self._residual, eta=y0, t0=t0, h=h, ny=ny, nw=nw)
        jac = jax.jacfwd(residual)

        def cond_func(carry):
            _, it, err = carry
            return jnp.logical_and(it < self.newton_maxiter, err > self.newton_tol)

        def body_func(carry):
            x, it, _ = carry
            r = residual(x)
            J = jac(x)
            dx = jnp.linalg.solve(J, -r)
            x_new = x + dx
            return (x_new, it + 1, jnp.max(jnp.abs(dx)))
        
        init = (x0, 0, jnp.array(jnp.inf))
        x_sol, n_iter, err = jax.lax.while_loop(cond_func, body_func, init)

        Y = x_sol[: self.s * ny].reshape(self.s, ny)
        W = x_sol[self.s * ny :].reshape(self.s, nw)
        t_stages = t0 + self.c * h

        f_vmap = jax.vmap(self.f, in_axes=(0, 0, 0))
        F = f_vmap(Y, W, t_stages)

        y1 = y0 + h * (self.b @ F)
        ell = jnp.linalg.solve(self.A, W - w0[None, :]) / h
        w1 = w0 + h * (self.b @ ell)

        return y1, w1

    def integrate(self, y0: jax.Array, w0: jax.Array, t0: float, h: float, n: int):
        """
        Apply RungeKutta.step() n times via jax.lax.scan.

        Returns full trajectories including initial conditions as first row.
        """

        def scan_func(carry, _):
            y, w, t = carry
            y_new, w_new = self.step(y, w, t, h)
            return (y_new, w_new, t+h), (y_new, w_new)

        (_, _, _), (ys, ws) = jax.lax.scan(scan_func, (y0, w0, t0), None, length=n)
        ys = jnp.concatenate([y0[None, :], ys], axis=0)
        ws = jnp.concatenate([w0[None, :], ws], axis=0)
        return ys, ws

    def solve(self, y0: jax.Array, w0: jax.Array, t0: float, T: float, h: float):
        n_full = int((T - t0) // h)
        ys, ws = self.integrate(y0, w0, t0, h, n_full)

        t_reached = t0 + n_full * h
        remainder = T - t_reached

        if remainder > 1e-12:
            y_last, w_last = ys[-1], ws[-1]
            y_final, w_final = self.step(y_last, w_last, t_reached, remainder)
            ys = jnp.concatenate([ys, y_final[None, :]], axis=0)
            ws = jnp.concatenate([ws, w_final[None, :]], axis=0)
            t_grid = jnp.concatenate([
                t0 + jnp.arange(n_full + 1) * h,
                jnp.array([T]),
            ])
        else:
            t_grid = t0 + jnp.arange(n_full + 1) * h

        return t_grid, ys, ws

class PartitionedRungeKutta:
    def __init__(self, As: Sequence[jax.Array], b: jax.Array, c: jax.Array, f, g, phi, newton_tol: float = 1e-6, newton_maxiter: int = 40):
        As = tuple(jnp.asarray(A) for A in As)
        self.s = len(b)

        for A in As:
            assert A.shape == (self.s, self.s), "All tableaus must have the same number of stages."

        self.A = As[0]
        self.A_hat = As[1]
        self.b = b
        self.c = c
        self.f = f
        self.g = g
        self.phi = phi
        self.newton_tol = newton_tol
        self.newton_maxiter = newton_maxiter

    def _assemble(self, unknowns, u0, lambda0, nu, nv, nlambda):
        """
        Assembles the vector of unknowns into the stage variables. They are assembled as
        U_2, ..., U_s, V_1, ..., V_s, Lambda_2, ..., Lambda_s. U_1 and Lambda_1 are given by the initial conditions u0 and lambda0.
        """
        s = self.s
        Us = unknowns[: (s-1) * nu].reshape(s - 1, nu)
        rest = unknowns[(s-1) * nu :]
        V = rest[: s * nv].reshape(s, nv)
        Lambdas = rest[s * nv :].reshape(s - 1, nlambda)
        U = jnp.concatenate([u0[None, :], Us], axis=0)
        Lambda = jnp.concatenate([lambda0[None, :], Lambdas], axis=0)
        return U, V, Lambda

    def _residual(self, unknowns, u0, v0, lambda0, t0, h, nu, nv, nlambda):
        U, V, Lambda = self._assemble(unknowns, u0, lambda0, nu, nv, nlambda)
        t_stages = t0 + self.c * h

        f_vmap = jax.vmap(self.f, in_axes=(0, 0, 0))
        g_vmap = jax.vmap(self.g, in_axes=(0, 0, 0, 0))
        phi_vmap = jax.vmap(self.phi, in_axes=(0, 0, 0))

        F = f_vmap(U, V, t_stages)
        G = g_vmap(U, V, Lambda, t_stages)

        R_U = U - u0[None, :] - h * (self.A @ F)
        R_V = V - v0[None, :] - h * (self.A_hat @ G)

        v_stages = v0[None, :] + h * (self.A @ G)
        R_phi = phi_vmap(U, v_stages, t_stages)

        R_U = R_U[1:]
        R_phi = R_phi[1:]

        return jnp.concatenate([R_U.reshape(-1), R_V.reshape(-1), R_phi.reshape(-1)])

    def step(self, u0: jax.Array, v0: jax.Array, lambda0: jax.Array, t0: float, h: float):
        nu = u0.shape[0]
        nv = v0.shape[0]
        nlambda = lambda0.shape[0]
        s = self.s

        Ur0 = jnp.tile(u0, (s - 1, 1))
        V0 = jnp.tile(v0, (s, 1))
        Lamb0 = jnp.tile(lambda0, (s - 1, 1))
        x0 = jnp.concatenate([Ur0.reshape(-1), V0.reshape(-1), Lamb0.reshape(-1)])

        residual = partial(self._residual, u0=u0, v0=v0, lambda0=lambda0, t0=t0, h=h, nu=nu, nv=nv, nlambda=nlambda)
        jac = jax.jacfwd(residual)

        def cond_func(carry):
            _, it, err = carry
            return jnp.logical_and(it < self.newton_maxiter, err > self.newton_tol)

        def body_func(carry):
            x, it, _ = carry
            r = residual(x)
            J = jac(x)
            dx = jnp.linalg.solve(J, -r)
            x_new = x + dx
            return (x_new, it + 1, jnp.max(jnp.abs(dx)))

        init = (x0, 0, jnp.array(jnp.inf))
        x_sol, n_iter, err = jax.lax.while_loop(cond_func, body_func, init)

        U, V, Lambda = self._assemble(x_sol, u0, lambda0, nu, nv, nlambda)
        t_stages = t0 + self.c * h

        f_vmap = jax.vmap(self.f, in_axes=(0, 0, 0))
        g_vmap = jax.vmap(self.g, in_axes=(0, 0, 0, 0))
        F = f_vmap(U, V, t_stages)
        G = g_vmap(U, V, Lambda, t_stages)

        u1 = u0 + h * (self.b @ F)
        v1 = v0 + h * (self.b @ G)
        lam1 = Lambda[-1]

        return u1, v1, lam1

    def integrate(self, u0: jax.Array, v0: jax.Array, lambda0: jax.Array, t0: float, h: float, n: int):
        def scan_func(carry, _):
            u, v, lam, t = carry
            u1, v1, lam1 = self.step(u, v, lam, t, h)
            return (u1, v1, lam1, t + h), (u1, v1, lam1)

        (_, _, _, _), (us, vs, lams) = jax.lax.scan(scan_func, (u0, v0, lambda0, t0), None, length=n)
        us = jnp.concatenate([u0[None, :], us], axis=0)
        vs = jnp.concatenate([v0[None, :], vs], axis=0)
        lams = jnp.concatenate([lambda0[None, :], lams], axis=0)
        return us, vs, lams

    def solve(self, u0: jax.Array, v0: jax.Array, lambda0: jax.Array, t0: float, T: float, h: float):
        n_full = int((T - t0) // h)
        ts = t0 + jnp.arange(n_full + 1) * h
        us, vs, lams = self.integrate(u0, v0, lambda0, t0, h, n_full)
        return ts, us, vs, lams