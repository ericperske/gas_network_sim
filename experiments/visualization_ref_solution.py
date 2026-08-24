from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from gasnetwork.topology import Node, Pipe
from gasnetwork.model import build_node_split
from utils import load_reference_solution, downsample_by_time

"""
Loads the numerical reference solution (as produced by network_reference_solution.py)
and plots all its components over time: densities (rho), fluxes (q) and the
node enthalpies (z), each in their own subplot of one combined figure.
"""

REF_SOLUTION_PATH = Path(__file__).resolve().parent / "ref_solution_T=100.npz"

# ====================== network topology (for labeling only) ======================
# Mirrors the network built in network_reference_solution.py; not re-simulated here,
# only used to label which rho/q/z component belongs to which pipe/node.
nodes = [
    Node(id=1, h=1.0),      # enthalpy boundary node
    Node(id=2),              # interior node
    Node(id=3, flux=0.0)     # flux boundary node
]

pipes = [
    Pipe(id=1, innode=nodes[0], outnode=nodes[1], length=3500.0, flux_refinement=2),
    Pipe(id=2, innode=nodes[1], outnode=nodes[2], length=3500.0, flux_refinement=2),
]

rho_labels = [f"pipe {pipe.id}, elem {k}" for pipe in pipes for k in range(pipe.n_elements)]
q_labels = [f"pipe {pipe.id}, node {k}" for pipe in pipes for k in range(pipe.n_fluxes)]

split = build_node_split(nodes)
z_labels = [f"node {nodes[i].id}" for i in split.given_q_idx]

# ====================== load reference solution ======================
ref = load_reference_solution(REF_SOLUTION_PATH)
t, rho, q, z = downsample_by_time(ref["t"], ref["rho"], ref["q"], ref["z"], dt=0.1)

# ====================== plot rho, q and z in one combined figure ======================
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
ax_z.set_ylabel("h")
ax_z.set_xlabel("t")
ax_z.set_title("Enthalpies")
ax_z.grid(True, ls="--", alpha=0.5)
ax_z.legend()

fig.suptitle("Reference solution")
fig.tight_layout()
plt.show()
