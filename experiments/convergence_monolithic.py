import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from gasnetwork.runge_kutta import RungeKutta
from gasnetwork.coefficients import get_tableau
from utils import plot_convergence, empirical_order, draw_slope_triangle

"""
Convergence test for the (monolithic) Runge-Kutta implementation on a simple
index-2 DAE in port-Hamiltonian form with a known analytic solution. The test
DAE is given by

    y1' = w
    y2' = sin(t) - y2
    0   = y1 - cos(t)

Exact solution: y1(t) = cos(t),
                y2(t) = 0.5 * exp(-t) + 0.5 * (sin(t) - cos(t)),
                w(t)  = -sin(t)

y2 is the unconstrained component and is therefore used for showing the
convergence order of the method in the y-component. y1 is constrained by the
algebraic equation and stays exact up to numerical precision, so it does not
show the convergence order of the method.

The convergence order is expected to be O(h^p) for y2 and O(h^q) for w, where
p and q are the parameters of the Radau IIA method satisfying the Runge-Kutta
order conditions B(p) and C(q). 
Those rates exceed our theoretical rates, cf. Ch. VII.4 and VII.5 of https://doi.org/10.1007/978-3-642-05221-7
"""

# ---------------- test problem ----------------
def f(y, w, t):
    return jnp.array([
        w[0],
        jnp.sin(t) - y[1],
    ])

def g(y, t):
    return jnp.array([y[0] - jnp.cos(t)])

Y0 = jnp.array([1.0, 0.0])
W0 = jnp.array([0.0])
T_END = 60.0

def y2_exact(t):
    return 0.5 * jnp.exp(-t) + 0.5 * (jnp.sin(t) - jnp.cos(t))

def w_exact(t):
    return -jnp.sin(t)

EXPECTED_ORDERS = {
    1: {"y2": 1.0, "w": 1.0},
    2: {"y2": 3.0, "w": 2.0},
    3: {"y2": 5.0, "w": 3.0},
}


def run_convergence_toy_dae(s: int, h_values: tuple[float, ...]):
    tableau = get_tableau(family="radau_iia", s=s)
    rk = RungeKutta(tableau.A, tableau.b, tableau.c, f, g, newton_tol=1e-14, newton_maxiter=60)

    hs = np.array(h_values)
    errs_y2 = np.empty(len(h_values))
    errs_w = np.empty(len(h_values))
    for i, h in enumerate(h_values):
        n = round(T_END / h)
        ys, ws = rk.integrate(Y0, W0, 0.0, h, n)
        errs_y2[i] = float(jnp.abs(ys[-1, 1] - y2_exact(T_END)))
        errs_w[i] = float(jnp.abs(ws[-1, 0] - w_exact(T_END)))

    return hs, {"y2": errs_y2, "w": errs_w}


# ====================== run convergence study ======================
# s=3 (order 5 for y2) converges too fast, so the rates are tested with
# larger values for h.
stage_hs = {
    1: (0.1, 0.05, 0.025, 0.0125, 0.00625),
    2: (0.1, 0.05, 0.025, 0.0125, 0.00625),
    3: (0.2, 0.1, 0.05, 0.025),
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
                         title="Monolithic RK convergence on test DAE")  # type: ignore

triangle_positions = {
    "y2": {1: (0.5, -0.4), 3: (0.5, 3.0), 5: (0.2, 5.5)},
    "w": {1: (0.5, -0.4), 2: (0.5, 1.2), 3: (0.2, 3.2)},
}
for ax, (var_name, series) in zip(axes, sol_dict.items()):  # type: ignore
    all_h = np.concatenate([h for h, _ in series.values()])
    all_err = np.concatenate([err for _, err in series.values()])
    h_range = (all_h.max(), all_h.min())
    y_range = (all_err.max(), all_err.min())
    for order, (h_frac, y_decades) in triangle_positions[var_name].items():
        draw_slope_triangle(ax, h_range=h_range, y_range=y_range, h_frac=h_frac, y_decades_below_max=y_decades, order=order)
    ax.legend()

plt.show()
