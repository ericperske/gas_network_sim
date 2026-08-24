# gas_network_sim
```bash
gas_network_sim/
├── gasnetwork/                  # Main package for gas network simulation
│ ├── __init__.py                # Package initialization
│ ├── basis_functions.py         # Spatial basis functions (piecewise constant / linear)
│ ├── coefficients.py            # Butcher tableaus (Radau IIA, Lobatto IIIA/IIIB/IIIC)
│ ├── constants.py               # Physical constants
│ ├── equations.py               # f/g/phi right-hand sides and thermodynamic relations
│ ├── errors.py                  # Custom errors for gas_network_sim
│ ├── model.py                   # Port-Hamiltonian model assembly (E, J, R, B, z)
│ ├── runge_kutta.py             # Runge-Kutta and Partitioned Runge-Kutta methods
│ ├── solver.py                  # GasNetworkSim: builds and runs a network simulation
│ └── topology.py                # Node / Pipe network topology
├── tests/                       # Unit tests
│ ├── test_equations.py          # Tests for the f/g/phi equations
│ ├── test_models.py             # Tests for model assembly
│ ├── test_pipes.py              # Tests for pipe components
│ └── test_solver_incidence.py   # Tests for generation of the incidence matrix
├── experiments/                 # Numerical experiments (see "Running the experiments" below)
├── README.md                    # Project documentation
├── requirements.txt             # Required Python packages
└── pyproject.toml               # Package metadata / build configuration
```

This repository provides a free-to-use solver for the simulation of gasnetworks, developed as part of my master's thesis. 

The mathematical formulation of the gas network and it's components is based on the ISO-1 formulation of pipes. This form provides a system of partial Differential-Algebraic Equations (pDAEs), which lead, after discretization in space, to a system of Differential-Algebraic Equations (DAEs) of index 2 in port-hamiltonian form, i.e.,
```math
\begin{aligned}
\dot{y}&= f(y,w,t), \\
0&= g(y,t).
\end{aligned}
```

This system is then solved using stiffly-accurate Runge-Kutta-methods. Here, we chose Radau IIA and Lobatto IIIC.

The system can also be formulated in a different manner to apply a partitioned Runge-Kutta method instead. For this reformulation, see https://doi.org/10.1007/s10543-021-00871-2.
The reformulated system is obtained by defining projector ```math P``` and ```math Q``` and reads
```math
\begin{aligned}
\dot{u} &= f(u,v,t) \\
\dot{v} &= g(u,v,\lambda,t) \\
0 &= \phi(u,v,t)
\end{aligned}
```
It is solved using the Lobatto IIIA-IIIB method.

## Naming convention
Two equivalent sets of variable names are used across the code and experiments, depending on whether the monolithic or the partitioned formulation above is used:

| Formulation | State | Constraint / algebraic variable | Method | Butcher tableau |
| --- | --- | --- | --- | --- |
| Monolithic (`RungeKutta`) | `y` (or `rho`/`q` when split into density/flux blocks) | `w` | `gasnetwork.runge_kutta.RungeKutta` | Radau IIA or Lobatto IIIC |
| Partitioned (`PartitionedRungeKutta`) | `u` (density-like) and `v` (flux-like) | `lambda`/`lam` | `gasnetwork.runge_kutta.PartitionedRungeKutta` | Lobatto IIIA/IIIB pair |

`rho`/`q` (network) refer to the same roles as `y` (toy DAEs), just named after the physical quantity (density/flux) instead of the abstract state when the experiment concerns the full pipe network rather than a toy problem; `u`/`v` are the corresponding partitioned-state names in both cases. `q1` and `h2` denote the two boundary-condition profiles of a node (a given flux and a given enthalpy respectively) and are unrelated to the `u`/`v` state names above; `s` denotes the number of Runge-Kutta stages, `h` the step size.

## Usage
In order to use this repository for simulations start by installing all required packages using\
```pip install -r requirements.txt```

## Running the experiments
All numerical experiments live in [experiments/](experiments/) and are run as plain scripts. Each script prints its observed convergence orders / diagnostics to the console and opens matplotlib figures (some also save `.png` files to `experiments/figures/`). Run them from the repository root, e.g.:
```bash
python experiments/convergence_monolithic.py
```

### 1. Toy-model convergence studies
These validate the Runge-Kutta methods against a toy DAE with a known closed-form solution, independent of the gas network model.

- `convergence_monolithic.py`: convergence of `RungeKutta` (Radau IIA, s=1,2,3) on a toy index-2 DAE.
- `convergence_partitioned.py`: convergence of `PartitionedRungeKutta` (Lobatto IIIA/IIIB, s=2,3,4) on a toy index-2 DAE.

Both are self-contained and can be run directly with no prior setup.

### 2. Gas-network convergence studies
These validate the Runge-Kutta methods on the actual 2-pipe gas network model, against a high-resolution numerical reference solution.

- `convergence_2_pipe_network_monolithic_radauiia.py`: `RungeKutta` with Radau IIA (s=1,2,3).
- `convergence_2_pipe_network_monolithic_lobattoiiic.py`: `RungeKutta` with Lobatto IIIC (s=2,3,4).
- `convergence_2_pipe_network_partitioned.py`: `PartitionedRungeKutta` with Lobatto IIIA/IIIB (s=2,3,4).
- `convergence_2_pipe_network_partitioned.py`: `PartitionedRungeKutta` with Lobatto IIIA/IIIB (s=2,3,4), but under artifical defect on initial values.

These three scripts load `experiments/ref_solution.npz` (already committed to the repo) as the reference solution, so they can be run directly as well. If you need to regenerate that reference solution (e.g. after changing the network scenario), run:
```bash
python experiments/network_reference_solution.py
```
Note: this script currently saves its output under the name `ref_solution_after_initial_data_fix.npz` — rename (or copy) the resulting file to `experiments/ref_solution.npz` before rerunning the convergence studies above.

### 3. Perturbation experiments
`perturbation_experiments.py` is a defect-injection study comparing how the partitioned method (PRK) and the monolithic methods (Radau IIA, Lobatto IIIC) handle a controlled constraint defect `theta_0` in the initial data. It reuses `experiments/ref_solution.npz` as well, and caches each individual run under `experiments/perturbation_solutions/` (delete that directory, or set `FORCE_RERUN_RUNS = True` in the script, to force a full recompute). Run with:
```bash
python experiments/perturbation_experiments.py
```
This produces `experiments/perturbation_solutions/summary.csv` and `P7_middle_link_table.csv`, plus six figures `perturbation_P1_...png` through `perturbation_P6_...png` in `experiments/figures/`.
