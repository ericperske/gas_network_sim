import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from pathlib import Path
import matplotlib.pyplot as plt

from gasnetwork.topology import Node, Pipe
from gasnetwork.equations import boundary_enthalpy, pressure_from_enthalpy
from gasnetwork.coefficients import get_tableau
from utils import load_reference_solution, run_convergence_study_network, plot_convergence, empirical_order, draw_slope_triangle

"""
This python file is used to test the implementation of the monolithic Runge-Kutta method against
the numerical reference solution.
"""

REF_SOLUTION_PATH = Path(__file__).resolve().parent / "ref_solution.npz"

# ====================== building the network ======================
ALPHA = 5.9 * 1e-3
SUPPLY_BAR = 80.0
DEMAND_PLATEAU = 0.8

H_START = 1.0
H_END = boundary_enthalpy(SUPPLY_BAR, DEMAND_PLATEAU, ALPHA)  # "the real supply" value

# Pressure whose (zero-flow) boundary enthalpy equals H_START, so the initial density
# is consistent with h2_profile(0) = H_START instead of the final supply pressure.
SUPPLY_BAR_START = pressure_from_enthalpy(H_START, 0.0, ALPHA)

nodes = [
    Node(id=1, h=H_END),    # placeholder static value; the actual (time-varying) BC
                             # used in the simulation is h2_profile(t) below.
    Node(id=2),              # interior node
    Node(id=3, flux=0.0)     # flux node; demand ramps up over time (see demand_profile)
]

pipes = [
    Pipe(id=1, innode=nodes[0], outnode=nodes[1], length=3500.0, flux_refinement=2),
    Pipe(id=2, innode=nodes[1], outnode=nodes[2], length=3500.0, flux_refinement=2),
]

# ====================== boundary condition ramps ======================
def demand_profile(t: float) -> jnp.ndarray:
    t_mid = 25.0                      # midpoint of the [15, 35] ramp
    width = (35.0 - 15.0) / 6.0       # controls transition steepness
    node3_demand = DEMAND_PLATEAU * 0.5 * (1.0 + jnp.tanh((t - t_mid) / width))
    return jnp.array([0.0, -node3_demand])
# the demand is not a real world scenario, but rather a demand that was verified numerically
# to create a steady state solution with the given physics of the network

def h2_profile(t: float) -> jnp.ndarray:
    t_mid = 20.0                      # midpoint of the [10, 30] ramp
    width = (30.0 - 10.0) / 6.0       # controls transition steepness
    h_val = H_START + (H_END - H_START) * 0.5 * (1.0 + jnp.tanh((t - t_mid) / width))
    return jnp.array([h_val])


# ====================== load the monolithic reference solution ======================
ref = load_reference_solution(REF_SOLUTION_PATH)
y_ref = jnp.concatenate([ref["rho"], ref["q"]], axis=-1)
w_ref = ref["z"]

# ====================== run convergence study ======================
t0 = 0.0
T = 40.0
h_values = [2 ** (-k) for k in range(0, 4)]
stage_tols = {2: 1e-14, 3: 1e-14, 4: 1e-14}

runs = {}
for s, tol in stage_tols.items():
    tableau = get_tableau(family="lobatto_iiic", s=s)

    runs[s] = run_convergence_study_network(
        nodes=nodes, pipes=pipes, method="RungeKutta",
        A=tableau.A, b=tableau.b, c=tableau.c,
        h_values=h_values, reference=ref, t0=t0, T=T, q1=demand_profile, h2=h2_profile,
        SUPPLY_BAR=SUPPLY_BAR_START, ALPHA=ALPHA, tol=tol,
    )

# ====================== empirical order (theoretical: s=1 -> 1, s=2 -> 3, s=3 -> 5) ======================
for s, per_var in runs.items():
    for var_name, (h, err) in per_var.items():
        print(f"Lobatto IIIC, s={s}, {var_name} observed order:", empirical_order(h, err))

# ====================== assemble per-variable series and plot ======================
COMBINED_PLOT = False
sol_dict = {}
for s, per_var in runs.items():
    for var_name, (h, err) in per_var.items():
        sol_dict.setdefault(var_name, {})[f"s={s}"] = (h, err)

axes = plot_convergence(sol_dict, combined=COMBINED_PLOT, title="Monolithic convergence for the 2-pipe network")

triangle_positions = {
    "y": {2: (0.5, -0.7), 4: (0.5, 3.5), 6: (0.125, 6.5)},
    "w": {1: (0.5, -0.1), 2: (0.5, 1.8), 3: (0.5, 3.3)},
}

for ax, (var_name, series) in zip(axes, sol_dict.items()):  # type: ignore
    all_h = jnp.concatenate([h for h, _ in series.values()])
    all_err = jnp.concatenate([err for _, err in series.values()])
    h_range = (float(all_h.max()), float(all_h.min()))
    y_range = (float(all_err.max()), float(all_err.min()))
    for order, (h_frac, y_decades) in triangle_positions[var_name].items():
        draw_slope_triangle(ax, h_range=h_range, y_range=y_range, h_frac=h_frac, y_decades_below_max=y_decades, order=order)


plt.show()