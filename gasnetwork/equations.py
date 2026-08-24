import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from time import time
from scipy.optimize import brentq

from gasnetwork.model import ModelParams
from gasnetwork.constants import GasConstants

# ========================= Utility functions =========================
def pressure_to_density(P_bar: float, alpha: float, consts=GasConstants) -> float:
    """
    Uses the equation of state to get the direct relationship
    between pressure and density.
    """
    RT = (consts.R / consts.M) / consts.PA_PER_BAR * consts.T
    return P_bar / (RT * (1 + alpha * P_bar))


def boundary_enthalpy(P_bar: float, q_ref: float, alpha: float, consts=GasConstants) -> float:
    """
    Same kinetic + pressure formula z_h uses (see z_pip.build), evaluated at the
    density implied by P_bar and a reference flux q_ref, so h_2 sits on the same
    scale as the pipe's own z_h instead of being an unadjusted number.
    """
    RT = (consts.R / consts.M) / consts.PA_PER_BAR * consts.T
    rho = pressure_to_density(P_bar, alpha, consts)
    P_prime = RT * float(jnp.log((rho * (1 - alpha * RT)) / (1 - alpha * RT * rho))) + RT / (1 - alpha * RT * rho)
    kinetic = q_ref**2 / (2 * rho**2)
    return kinetic + P_prime


def pressure_from_enthalpy(h_target: float, q_ref: float, alpha: float, consts=GasConstants,
                            bracket: tuple[float, float] = (1e-6, 500.0)) -> float:
    """
    Inverts boundary_enthalpy: finds the supply pressure P_bar whose boundary
    enthalpy (at the same reference flux q_ref) equals h_target. boundary_enthalpy
    has no closed-form inverse, so this fiunds it numerically instead. Used to pick an
    initial density consistent with a chosen starting enthalpy (e.g. a scenario's
    H_START), rather than with the network's final/target supply pressure.
    """
    return brentq(lambda P_bar: boundary_enthalpy(P_bar, q_ref, alpha, consts) - h_target, *bracket) # type: ignore


# ========================= functions for gasnetworks =========================
# All functions below act only on the globally assembled J, E(c), R(c), z(c), B_1,
# B_2 (built per-pipe over each pipe's own flux_refinement in model.py) and on
# n_rho = u.shape[0], the total density count across all pipes. None of them
# index into pipe-local structure, so they work unchanged for any (and mixed)
# per-pipe flux_refinement values.

def f_rk(y: jnp.ndarray, w: jnp.ndarray, t: float, p: ModelParams) -> jnp.ndarray:
    """
    y_dot = E(c)^{-1} ((J - R(c)) z(c) + B1 w + B2 h2(t))
    """
    z = p.z_func(y)
    rhs = (p.J - p.R(y)) @ z + p.B_1 @ w + p.B_2 @ p.h_2(t)
    return jnp.linalg.solve(p.E(y), rhs)

def g_rk(y: jnp.ndarray, t: float, p: ModelParams) -> jnp.ndarray:
    """
    0 = g(y) = q_1(t) - B_1^T z(y)
    """
    z = p.z_func(y)
    return p.q_1(t) - p.B_1.T @ z


def f_prk(u: jnp.ndarray, v: jnp.ndarray, t: float, p: ModelParams) -> jnp.ndarray:
    """
    du/dt = E_rho_ma^{-1} * ((J - R(c)) z(c))_rho ,  c = [u, v]
    The rho block of f; B_1/B_2 don't appear here since B is zero on
    the rho rows, so h_1 (the multiplier) never enters this equation.
    """
    n_rho = u.shape[0]
    c = jnp.concatenate([u, v])
    z = p.z_func(c)
    rhs_rho = ((p.J - p.R(c)) @ z)[:n_rho]
    E_rho_ma = p.E(c)[:n_rho, :n_rho]
    return jnp.linalg.solve(E_rho_ma, rhs_rho)

def g_prk(u: jnp.ndarray, v: jnp.ndarray, lam: jnp.ndarray, t: float, p: ModelParams) -> jnp.ndarray:
    """
        dv/dt = E_q_mo(c)^{-1} * (
            ((J - R(c)) z(c))_q + B_1_q @ lam + B_2_q @ h_2 - E_rho_mo(c) @ f_prk(u, v, t)
        ),  c = [u, v]
        """
    n_rho = u.shape[0]
    c = jnp.concatenate([u, v])
    z = p.z_func(c)
    rhs_q = ((p.J - p.R(c)) @ z)[n_rho:] + p.B_1[n_rho:, :] @ lam + p.B_2[n_rho:, :] @ p.h_2(t)
    E_full = p.E(c)
    E_rho_mo = E_full[n_rho:, :n_rho]
    E_q_mo = E_full[n_rho:, n_rho:]
    rhs_q = rhs_q - E_rho_mo @ f_prk(u, v, t, p)
    return jnp.linalg.solve(E_q_mo, rhs_q)

def phi_prk(u: jnp.ndarray, v: jnp.ndarray, t: float, p: ModelParams) -> jnp.ndarray:
    """
        Algebraic constraint, identical to g(c, t, p): 0 = q_1(t) - B_1^T z(c)
        """
    c = jnp.concatenate([u, v])
    return g_rk(c, t, p)