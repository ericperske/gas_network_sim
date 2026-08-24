import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from gasnetwork.runge_kutta import PartitionedRungeKutta
from gasnetwork.coefficients import get_tableau
from utils import plot_convergence, empirical_order, draw_slope_triangle

"""
Convergence test for the partitioned Runge-Kutta implementation, as described in
https://arxiv.org/abs/1810.10298v1. Checks convergence of the Lobatto IIIA/IIIB
pair on a toy DAE with a known analytic solution, used as the reference instead
of a numerical one.

The test DAE is given by

                   -v = f(u, v, t)
      sin(t) - lambda = g(u, v, lam, t)
   v + u + 2 * cos(t) = phi(u, v, t)

The system has the exact solution u(t) = sin(t) - cos(t), v(t) = -sin(t) - cos(t),
lam(t) = cos(t).

The convergence order being tested is according to Cor. 3.7.1 of the referenced
source, which states for the Lobatto IIIA/IIIB pair with s stages:
    u_n - u(t_n) = O(h^(2s-2))
    v_n - v(t_n) = O(h^(2s-2))
    lam_n - lam(t_n) = O(h^s) (s even) or O(h^(s-1)) (s odd)
"""

# ---------------- test problem ----------------
def f(u, v, t):
    return -v

def g(u, v, lam, t):
    return jnp.sin(t) - lam

def phi(u, v, t):
    return v + u + 2 * jnp.cos(t)

U0 = jnp.array([-1.0])      # sin(0) - cos(0)
V0 = jnp.array([-1.0])      # -sin(0) - cos(0)
LAM0 = jnp.array([1.0])     # cos(0)
T_END = 1.0

def u_exact(t):
    return jnp.sin(t) - jnp.cos(t)

def v_exact(t):
    return -jnp.sin(t) - jnp.cos(t)

def lam_exact(t):
    return jnp.cos(t)

EXPECTED_ORDERS = {
    2: {"u": 2.0, "v": 2.0, "lambda": 2.0},
    3: {"u": 4.0, "v": 4.0, "lambda": 2.0},
    4: {"u": 6.0, "v": 6.0, "lambda": 4.0},
}


def run_convergence_toy_dae(s: int, h_values: tuple[float, ...]):
    iiia = get_tableau(family="lobatto_iiia", s=s)
    iiib = get_tableau(family="lobatto_iiib", s=s)
    rk = PartitionedRungeKutta([iiia.A, iiib.A], iiia.b, iiia.c, f, g, phi, newton_tol=1e-14, newton_maxiter=60)

    hs = np.array(h_values)
    errs_u = np.empty(len(h_values))
    errs_v = np.empty(len(h_values))
    errs_lam = np.empty(len(h_values))
    for i, h in enumerate(h_values):
        n = round(T_END / h)
        us, vs, lams = rk.integrate(U0, V0, LAM0, 0.0, h, n)
        errs_u[i] = float(jnp.abs(us[-1, 0] - u_exact(T_END)))
        errs_v[i] = float(jnp.abs(vs[-1, 0] - v_exact(T_END)))
        errs_lam[i] = float(jnp.abs(lams[-1, 0] - lam_exact(T_END)))

    return hs, {"u": errs_u, "v": errs_v, "lambda": errs_lam}


# ====================== run convergence study ======================
# s=4 (order 6 for u,v) converges too fast, so the rates are tested with
# larger values for h.
stage_hs = {
    2: (0.1, 0.05, 0.025, 0.0125, 0.00625),
    3: (0.1, 0.05, 0.025, 0.0125, 0.00625),
    4: (0.1, 0.05, 0.025),
}

sol_dict = {}
for s, h_values in stage_hs.items():
    hs, errs = run_convergence_toy_dae(s, h_values)
    for var_name, err in errs.items():
        sol_dict.setdefault(var_name, {})[f"s={s}"] = (hs, err)
        print(f"s={s}, {var_name} observed order:", empirical_order(hs, err), # type: ignore
              " (expected:", EXPECTED_ORDERS[s][var_name], ")")

# ====================== assemble per-variable series and plot ======================
COMBINED_PLOT = False
axes = plot_convergence(sol_dict, combined=COMBINED_PLOT,
                         title="Partitioned RK convergence on test DAE")  # type: ignore

triangle_positions = {
    "u": {2: (0.55, -0.4), 4: (0.55, 5.0), 6: (0.15, 7.5)},
    "v": {2: (0.55, -0.4), 4: (0.55, 5.0), 6: (0.15, 7.5)},
    "lambda": {2: (0.55, -0.4), 4: (0.45, 5.0)},
}
for ax, (var_name, series) in zip(axes, sol_dict.items()): # type: ignore
    all_h = np.concatenate([h for h, _ in series.values()])
    all_err = np.concatenate([err for _, err in series.values()])
    h_range = (all_h.max(), all_h.min())
    y_range = (all_err.max(), all_err.min())
    for order, (h_frac, y_decades) in triangle_positions[var_name].items():
        draw_slope_triangle(ax, h_range=h_range, y_range=y_range, h_frac=h_frac, y_decades_below_max=y_decades, order=order)

plt.show()
