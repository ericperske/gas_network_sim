import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from pathlib import Path
import matplotlib.pyplot as plt

from gasnetwork.topology import Node, Pipe
from gasnetwork.equations import boundary_enthalpy, pressure_from_enthalpy
from gasnetwork.coefficients import get_tableau
from solver_perturbation import GasNetworkSimPerturbation
from utils import load_reference_solution, plot_convergence, empirical_order, draw_slope_triangle

"""
This python file is used to test the implementation of the partitioned Runge-Kutta method against
the numerical reference solution.

Unlike convergence_2_pipe_network_partitioned.py, this experiment injects an artificial constraint
defect of order theta_0 = 1e-6 into the initial data, using GasNetworkSimPerturbation (see
solver_perturbation.py): the initial v0 is pushed off the algebraic constraint manifold
phi(u0, v0, t0) = 0 so that ||phi(u0, v0, t0)||_2 = theta_0, exactly as in perturbation_experiments.py.
Since this defect does not shrink with the step size h, it is expected (per the theta_0/h scaling law
established there) to grow under refinement for the PRK's algebraic variable lambda, destroying its
convergence order.
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
    Node(id=1, h=H_END),     # placeholder static value; the actual (time-varying) BC
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
    """
    The demand profile is a C^infinity approximation of the one that was
    used originally.
    """
    t_mid = 25.0                      # midpoint of the [15, 35] ramp
    width = (35.0 - 15.0) / 6.0       # controls transition steepness
    node3_demand = DEMAND_PLATEAU * 0.5 * (1.0 + jnp.tanh((t - t_mid) / width))
    return jnp.array([0.0, -node3_demand])

def h2_profile(t: float) -> jnp.ndarray:
    t_mid = 20.0                      # midpoint of the [10, 30] ramp
    width = (30.0 - 10.0) / 6.0       # controls transition steepness
    h_val = H_START + (H_END - H_START) * 0.5 * (1.0 + jnp.tanh((t - t_mid) / width))
    return jnp.array([h_val])


# ====================== load the partitioned reference solution ======================
ref = load_reference_solution(REF_SOLUTION_PATH)
u_ref = ref["rho"]
v_ref = ref["q"]
lambda_ref = ref["z"]

# ====================== inject an artificial defect ======================
# THETA_0 is the L2 norm of the constraint defect phi(u0, v0, t0) injected into the initial
# data via GasNetworkSimPerturbation (see solver_perturbation.py), exactly as in
# perturbation_experiments.py. It does not shrink with h, so it should destroy the PRK's
# convergence order in lambda once the truncation error drops below the theta_0/h plateau.
THETA_0 = 1e-6

# ====================== convergence study (mirrors utils.run_convergence_study_network,
# but uses GasNetworkSimPerturbation with THETA_0 instead of GasNetworkSim) ======================
def run_convergence_study_network_perturbed(nodes, pipes, A, b, c, h_values, reference, t0, T,
                                              q1, h2, SUPPLY_BAR, ALPHA, tol, theta_0, maxiter=1000):
    ref_idx = int(T / reference["h_ref"])
    rho_ref = jnp.asarray(reference["rho"])[ref_idx]
    q_ref = jnp.asarray(reference["q"])[ref_idx]
    z_ref = jnp.asarray(reference["z"])[ref_idx]

    errors = {"rho": [], "q": [], "lambda": []}
    for h in h_values:
        sim = GasNetworkSimPerturbation(nodes=nodes, pipes=pipes, method="PartitionedRungeKutta",
                                         A=A, b=b, c=c, t0=t0, T=T, h=h, q1=q1, h2=h2, ALPHA=ALPHA)
        result = sim.simulate(SUPPLY_BAR=SUPPLY_BAR, tol=tol, maxiter=maxiter, theta_0=theta_0)  # type: ignore
        print("Current stepsize: ", h)
        rho_final, q_final, lambda_final = result[1][-1], result[2][-1], result[3][-1] # type: ignore
        errors["rho"].append(jnp.sqrt(jnp.sum((rho_final - rho_ref) ** 2)))
        errors["q"].append(jnp.sqrt(jnp.sum((q_final - q_ref) ** 2)))
        errors["lambda"].append(jnp.sqrt(jnp.sum((lambda_final - z_ref) ** 2)))

    h_array = jnp.asarray(h_values, dtype=float)
    return {var_name: (h_array, jnp.asarray(err)) for var_name, err in errors.items()}


# ====================== run convergence study ======================
t0 = 0.0
T = 40.0
h_values = [2 ** (-k) for k in range(-1, 3)]
stage_tols = {2: 1e-12, 3: 1e-12, 4: 1e-17}

# h_values = [2 ** (-k) for k in range(-1, 3)]
# stage_tols = {4: 1e-17}


runs = {}
for s, tol in stage_tols.items():
    lobatto_iiia = get_tableau(family="lobatto_iiia", s=s)
    lobatto_iiib = get_tableau(family="lobatto_iiib", s=s)
    runs[s] = run_convergence_study_network_perturbed(
        nodes=nodes, pipes=pipes,
        A=(lobatto_iiia.A, lobatto_iiib.A), b=lobatto_iiia.b, c=lobatto_iiia.c,
        h_values=h_values, reference=ref, t0=t0, T=T, q1=demand_profile, h2=h2_profile,
        SUPPLY_BAR=SUPPLY_BAR_START, ALPHA=ALPHA, tol=tol, theta_0=THETA_0,
    )

# ====================== empirical order (theoretical: s=2 -> 3, s=3 -> 5) ======================
for s, per_var in runs.items():
    for var_name, (h, err) in per_var.items():
        print(f"Lobatto IIIA/IIIB, s={s}, {var_name} observed order:", empirical_order(h, err))

# ====================== assemble per-variable series and plot ======================
COMBINED_PLOT = False
sol_dict = {}
for s, per_var in runs.items():
    for var_name, (h, err) in per_var.items():
        sol_dict.setdefault(var_name, {})[f"s={s}"] = (h, err)

axes = plot_convergence(sol_dict, combined=COMBINED_PLOT, title="PRK convergence (perturbed, theta_0=1e-6)")

plt.show()