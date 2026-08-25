import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from time import time

from gasnetwork.solver import GasNetworkSim
from gasnetwork.topology import Node, Pipe
from gasnetwork.equations import boundary_enthalpy, pressure_from_enthalpy
from gasnetwork.coefficients import get_tableau
from utils import save_ref_solution

"""
This python file creates the same exact network, that is used in the experiments.
It is solved with a much higher temporal resolution, in order to generate a 
numerical reference solution and test that against the same network structure
with lower temporal resolutions and differing spatial resolutions.
"""

# ====================== building the network ======================
ALPHA = 5.9 * 1e-3
SUPPLY_BAR = 80.0
DEMAND_PLATEAU = 0.8

H_START = 1.0
H_END = boundary_enthalpy(SUPPLY_BAR, DEMAND_PLATEAU, ALPHA)  # "the real supply" value

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

# ====================== define temporal resolution for ref solution ======================
t0 = 0.0
T = 40.0
h_ref = 2 ** (-17)

# ====================== run monolithic RK ======================
radau_iia = get_tableau(family="radau_iia", s=3)
toc_mono = time()
n_mono = GasNetworkSim(nodes=nodes, pipes=pipes, method="RungeKutta", A=radau_iia.A, b=radau_iia.b, c=radau_iia.c,
                            t0=t0, T=T, h=h_ref, q1=demand_profile, h2=h2_profile, ALPHA=ALPHA)

t_mono, y_mono, z_mono = n_mono.simulate(SUPPLY_BAR=SUPPLY_BAR_START, tol=1e-12) # type: ignore
tic_mono = time()
print(f"Done solving the monolithic model in {tic_mono-toc_mono:.3f} seconds.")
print()
print("Saving monolithic solution...")
save_ref_solution(
    name="ref_solution",
    t=t_mono,
    rho=y_mono[:, :sum(pipe.n_elements for pipe in pipes)],
    q=y_mono[:, sum(pipe.n_elements for pipe in pipes):],
    z=z_mono,
    h=h_ref
)
