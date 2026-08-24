from typing import Sequence, Optional

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from gasnetwork.topology import Node, Pipe
from gasnetwork.model import build_node_split
from utils import downsample_by_time

"""
Provides a reusable plot_solution() function that visualizes a (rho, q, z) solution
the same way visualization_ref_solution.py plots the reference solution: densities,
fluxes and node enthalpies/multipliers, each in their own subplot of one combined
figure.

As a usage example, this file also runs the partitioned Runge-Kutta (PRK) method on
the same 2-pipe network scenario used for the reference solution and displays the
result with plot_solution().
"""

# ====================== plotting function ======================
def plot_solution(t: jnp.ndarray, rho: jnp.ndarray, q: jnp.ndarray, z: jnp.ndarray,
                   nodes: Sequence[Node], pipes: Sequence[Pipe],
                   title: str = "Solution", z_ylabel: str = "h",
                   plot_dt: float = 0.1) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """
    Plots densities (rho), fluxes (q) and node enthalpies/multipliers (z) over time,
    each in their own subplot of one combined figure. Labels are derived from the
    given nodes/pipes, mirroring how visualization_ref_solution.py labels the
    reference solution's components. The series are downsampled to plot_dt spacing
    before plotting, so a finely-resolved solution (small step size h) doesn't bloat
    the plot with one point per solver step.
    """
    t, rho, q, z = downsample_by_time(t, rho, q, z, dt=plot_dt)

    rho_labels = [f"pipe {pipe.id}, elem {k}" for pipe in pipes for k in range(pipe.n_elements)]
    q_labels = [f"pipe {pipe.id}, node {k}" for pipe in pipes for k in range(pipe.n_fluxes)]

    split = build_node_split(nodes)
    z_labels = [f"node {nodes[i].id}" for i in split.given_q_idx]

    fig, (ax_rho, ax_q, ax_z) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))  # type: ignore

    for i, label in enumerate(rho_labels):
        ax_rho.plot(t, rho[:, i], label=label)
    ax_rho.set_ylabel(r"$\rho$")
    ax_rho.set_title("Densities")
    ax_rho.grid(True, ls="--", alpha=0.5)
    ax_rho.legend()

    for i, label in enumerate(q_labels):
        ax_q.plot(t, q[:, i], label=label)
    ax_q.set_ylabel("q")
    ax_q.set_title("Fluxes")
    ax_q.grid(True, ls="--", alpha=0.5)
    ax_q.legend()

    for i, label in enumerate(z_labels):
        ax_z.plot(t, z[:, i], label=label)
    ax_z.set_ylabel(z_ylabel)
    ax_z.set_xlabel("t")
    ax_z.set_title("Enthalpies")
    ax_z.grid(True, ls="--", alpha=0.5)
    ax_z.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax_rho, ax_q, ax_z)


# ====================== example: PRK solution on the reference-solution scenario ======================
if __name__ == "__main__":
    from gasnetwork.solver import GasNetworkSim
    from gasnetwork.equations import boundary_enthalpy, pressure_from_enthalpy
    from gasnetwork.coefficients import get_tableau

    # Mirrors the network and boundary conditions built in network_reference_solution.py
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

    def demand_profile(t: float) -> jnp.ndarray:
        t_mid = 25.0                      # midpoint of the [15, 35] ramp
        width = (35.0 - 15.0) / 6.0       # controls transition steepness
        node3_demand = DEMAND_PLATEAU * 0.5 * (1.0 + jnp.tanh((t - t_mid) / width))
        return jnp.array([0.0, -node3_demand])

    def h2_profile(t: float) -> jnp.ndarray:
        t_mid = 20.0                      # midpoint of the [10, 30] ramp
        width = (30.0 - 10.0) / 6.0       # controls transition steepness
        h_val = H_START + (H_END - H_START) * 0.5 * (1.0 + jnp.tanh((t - t_mid) / width))
        return jnp.array([h_val])

    t0 = 0.0
    T = 100.0
    h = 2 ** (-4)

    lobatto_iiia = get_tableau(family="lobatto_iiia", s=3)
    lobatto_iiib = get_tableau(family="lobatto_iiib", s=3)

    prk = GasNetworkSim(
        nodes=nodes, pipes=pipes, method="PartitionedRungeKutta",
        A=(lobatto_iiia.A, lobatto_iiib.A), b=lobatto_iiia.b, c=lobatto_iiia.c,
        t0=t0, T=T, h=h, q1=demand_profile, h2=h2_profile, ALPHA=ALPHA,
    )
    t_prk, rho_prk, q_prk, lambda_prk = prk.simulate(SUPPLY_BAR=SUPPLY_BAR_START, tol=1e-12)  # type: ignore

    plot_solution(t_prk, rho_prk, q_prk, lambda_prk, nodes=nodes, pipes=pipes,
                  title="Partitioned Runge-Kutta solution", z_ylabel=r"$\lambda$")
    plt.show()
