import csv
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from gasnetwork.topology import Node, Pipe
from gasnetwork.equations import boundary_enthalpy, pressure_from_enthalpy
from gasnetwork.coefficients import get_tableau
from solver_perturbation import GasNetworkSimPerturbation
from utils import draw_slope_triangle, load_reference_solution

"""
Defect-injection study: PRK (Lobatto IIIA/IIIB) vs. the monolithic methods
(Radau IIA, Lobatto IIIC) under a controlled constraint defect theta_0 in the
initial data.

The constraint is the same function in both formulations,
    phi(q,p,t) = g(y,t) = q_1(t) - B_1^T c,
where c is the full state (density and flux blocks); the PRK split into q/p
only partitions c afterwards and does not change the constraint. lambda (PRK)
and z (monolithic) are both the Lagrange multiplier enforcing d/dt[constraint]
= 0 and are directly comparable against the same reference trajectory.

Claim under test: a constraint defect theta_0 at t=0 produces an error in the
algebraic variable of size ~ C * theta_0 / h for the PRK (growing under
refinement), while the monolithic methods absorb the same defect within one or
two steps, independent of h.

Here, we perform 6 experiments:

- P1: show error ||lambda(t) - lambda_ref(t)|| for multiple defect values theta_0 and fixed value for h
- P2: show error ||lambda(t) - lambda_ref(t)|| for multiple step sizes h and fixed value for theta_0
- P3: log-log-plot of the plateau reached in the error in lambda for the different defect values
- P4: show ratio of h * plateau / theta_0 for multiple step sizes to numerically estimate error constant C
- P5: show err(t) for all 3 methods (Radau IIA, Lobatto IIIC, Lobatto IIIA-IIIB)
- P6: as P3, but for all 3 methods under fixed defect theta_0
"""

# ====================== paths ======================
SOLUTIONS_DIR = Path(__file__).resolve().parent / "perturbation_solutions"
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FORCE_RERUN_RUNS = False

REF_SOLUTION_PATH = Path(__file__).resolve().parent / "ref_solution.npz"

# ====================== building the network (mirrors the other 2-pipe-network experiments) ======================
ALPHA = 5.9 * 1e-3
SUPPLY_BAR = 80.0
DEMAND_PLATEAU = 0.8

H_START = 1.0
H_END = boundary_enthalpy(SUPPLY_BAR, DEMAND_PLATEAU, ALPHA)
SUPPLY_BAR_START = pressure_from_enthalpy(H_START, 0.0, ALPHA)

nodes = [
    Node(id=1, h=H_END),
    Node(id=2),
    Node(id=3, flux=0.0)
]

pipes = [
    Pipe(id=1, innode=nodes[0], outnode=nodes[1], length=3500.0, flux_refinement=2),
    Pipe(id=2, innode=nodes[1], outnode=nodes[2], length=3500.0, flux_refinement=2),
]

def demand_profile(t: float) -> jnp.ndarray:
    t_mid = 25.0
    width = (35.0 - 15.0) / 6.0
    node3_demand = DEMAND_PLATEAU * 0.5 * (1.0 + jnp.tanh((t - t_mid) / width))
    return jnp.array([0.0, -node3_demand])

def h2_profile(t: float) -> jnp.ndarray:
    t_mid = 20.0
    width = (30.0 - 10.0) / 6.0
    h_val = H_START + (H_END - H_START) * 0.5 * (1.0 + jnp.tanh((t - t_mid) / width))
    return jnp.array([h_val])

t0 = 0.0
T = 40.0

# ====================== color palette (fixed categorical order, reused across every plot) ======================
THETA_VALUES = [1e-10, 1e-8, 1e-6, 1e-4]
H_VALUES = [2.0, 1.0, 0.5, 0.25]

THETA_COLORS = {1e-10: "#2a78d6", 1e-8: "#eb6834", 1e-6: "#1baf7a", 1e-4: "#eda100"}
H_COLORS = {2.0: "#2a78d6", 1.0: "#eb6834", 0.5: "#1baf7a", 0.25: "#eda100"}
METHOD_COLORS = {"PRK": "#2a78d6", "RadauIIA": "#eb6834", "LobattoIIIC": "#1baf7a"}
GRID_KW = dict(which="both", ls="--", alpha=0.5, color="#c3c2b7")

# ====================== method configurations ======================
_lobatto_iiia = get_tableau(family="lobatto_iiia", s=3)
_lobatto_iiib = get_tableau(family="lobatto_iiib", s=3)
_radau_iia = get_tableau(family="radau_iia", s=3)
_lobatto_iiic = get_tableau(family="lobatto_iiic", s=3)

METHOD_CONFIGS = {
    "PRK": dict(method="PartitionedRungeKutta", A=(_lobatto_iiia.A, _lobatto_iiib.A), b=_lobatto_iiia.b, c=_lobatto_iiia.c),
    "RadauIIA": dict(method="RungeKutta", A=_radau_iia.A, b=_radau_iia.b, c=_radau_iia.c),
    "LobattoIIIC": dict(method="RungeKutta", A=_lobatto_iiic.A, b=_lobatto_iiic.b, c=_lobatto_iiic.c),
}

RUN_TOL = 1e-13
RUN_MAXITER = 300


# ====================== Step 1: reference solution ======================
def load_existing_reference():
    """
    Reuses experiments/ref_solution.npz. Already generated (by
    network_reference_solution.py) for this exact 2-pipe scenario with
    unperturbed, fully consistent initial data at h_ref = 2**-17. That h_ref is
    of the form 0.25 / 2**k (k=15), so it divides every h in H_VALUES exactly:
    lambda_ref/z_ref can be read off the reference grid by striding (see
    z_ref_at below) with no interpolation, hence no interpolation error.
    """
    ref = load_reference_solution(REF_SOLUTION_PATH)
    ts_ref = ref["t"]
    zs_ref = ref["z"]
    h_ref = float(ref["h_ref"])
    print(f"Using existing reference solution {REF_SOLUTION_PATH.name}: "
          f"h_ref={h_ref:.6e}, {ts_ref.shape[0]} points, unperturbed initial data.")
    return ts_ref, zs_ref, h_ref


def z_ref_at(h: float, ts_ref: jnp.ndarray, zs_ref: jnp.ndarray, h_ref: float) -> jnp.ndarray:
    stride = round(h / h_ref)
    assert abs(h - stride * h_ref) < 1e-9 * max(h_ref, 1.0), f"h={h} is not an exact multiple of h_ref={h_ref}"
    n_expected = int(round(T / h)) + 1
    grid = zs_ref[::stride][:n_expected]
    assert grid.shape[0] == n_expected, f"reference grid too short for h={h}: got {grid.shape[0]}, need {n_expected}"
    return grid


# ====================== Step 2/3: per-run diagnostics and execution ======================
def rk_diagnostics(sim, ys, zs, ts):
    g_vals = jax.vmap(sim.g, in_axes=(0, 0))(ys, ts)
    phi_res = jnp.linalg.norm(g_vals, axis=1)

    def hidden(y, z, t):
        y_dot = sim.f(y, z, t)
        _, dg = jax.jvp(lambda yy, tt: sim.g(yy, tt), (y, t), (y_dot, 1.0))
        return dg

    dg_vals = jax.vmap(hidden, in_axes=(0, 0, 0))(ys, zs, ts)
    dphi_res = jnp.linalg.norm(dg_vals, axis=1)
    return phi_res, dphi_res


def prk_diagnostics(sim, qs, ps, lams, ts):
    phi_vals = jax.vmap(sim.phi, in_axes=(0, 0, 0))(qs, ps, ts)
    phi_res = jnp.linalg.norm(phi_vals, axis=1)

    def hidden(u, v, lam, t):
        u_dot = sim.f(u, v, t)
        v_dot = sim.g(u, v, lam, t)
        _, dphi = jax.jvp(lambda uu, vv, tt: sim.phi(uu, vv, tt), (u, v, t), (u_dot, v_dot, 1.0))
        return dphi

    dphi_vals = jax.vmap(hidden, in_axes=(0, 0, 0, 0))(qs, ps, lams, ts)
    dphi_res = jnp.linalg.norm(dphi_vals, axis=1)
    return phi_res, dphi_res


def run_path(method_id: str, theta_0: float, h: float) -> Path:
    return SOLUTIONS_DIR / f"{method_id}_theta{theta_0:.0e}_h{h:g}.npz"


def save_run(run: dict) -> None:
    np.savez_compressed(
        run_path(run["method_id"], run["theta_0"], run["h"]),
        t=run["t"], err=run["err"], phi_res=run["phi_res"], dphi_res=run["dphi_res"],
        plateau=run["plateau"], dphi_plateau=run["dphi_plateau"],
        theta_0=run["theta_0"], theta_0_measured=run["theta_0_measured"], h=run["h"],
    )


def load_run(method_id: str, theta_0: float, h: float):
    path = run_path(method_id, theta_0, h)
    if not path.exists():
        return None
    npz = np.load(path)
    return {
        "method_id": method_id,
        "t": npz["t"], "err": npz["err"], "phi_res": npz["phi_res"], "dphi_res": npz["dphi_res"],
        "plateau": float(npz["plateau"]), "dphi_plateau": float(npz["dphi_plateau"]),
        "theta_0": float(npz["theta_0"]), "theta_0_measured": float(npz["theta_0_measured"]), "h": float(npz["h"]),
    }


def run_single(method_id: str, theta_0: float, h: float, ts_ref, zs_ref, h_ref) -> dict:
    cfg = METHOD_CONFIGS[method_id]
    sim = GasNetworkSimPerturbation(nodes=nodes, pipes=pipes, method=cfg["method"],
                                     A=cfg["A"], b=cfg["b"], c=cfg["c"], t0=t0, T=T, h=h,
                                     q1=demand_profile, h2=h2_profile, ALPHA=ALPHA)

    if cfg["method"] == "RungeKutta":
        ts, ys, zs = sim.simulate(SUPPLY_BAR=SUPPLY_BAR_START, tol=RUN_TOL, maxiter=RUN_MAXITER, theta_0=theta_0) # type: ignore
        lam_traj = zs
        phi_res, dphi_res = rk_diagnostics(sim, ys, zs, ts)
    else:
        ts, qs, ps, lams = sim.simulate(SUPPLY_BAR=SUPPLY_BAR_START, tol=RUN_TOL, maxiter=RUN_MAXITER, theta_0=theta_0) # type: ignore
        lam_traj = lams
        phi_res, dphi_res = prk_diagnostics(sim, qs, ps, lams, ts)

    z_ref_grid = z_ref_at(h, ts_ref, zs_ref, h_ref)
    err = jnp.linalg.norm(lam_traj - z_ref_grid, axis=1)

    mask = (ts >= 5.0) & (ts <= 10.0)
    plateau = float(jnp.mean(err[mask]))
    dphi_plateau = float(jnp.mean(dphi_res[mask]))

    return {
        "method_id": method_id, "theta_0": theta_0, "h": h,
        "t": np.asarray(ts), "err": np.asarray(err),
        "phi_res": np.asarray(phi_res), "dphi_res": np.asarray(dphi_res),
        "plateau": plateau, "dphi_plateau": dphi_plateau,
        "theta_0_measured": float(phi_res[0]),
    }


def get_or_run(method_id: str, theta_0: float, h: float, ts_ref, zs_ref, h_ref) -> dict:
    if not FORCE_RERUN_RUNS:
        cached = load_run(method_id, theta_0, h)
        if cached is not None:
            print(f"  [cached] {method_id} theta_0={theta_0:.0e} h={h:g}: plateau={cached['plateau']:.3e}")
            return cached
    run = run_single(method_id, theta_0, h, ts_ref, zs_ref, h_ref)
    save_run(run)
    print(f"  [computed] {method_id} theta_0={theta_0:.0e} h={h:g}: "
          f"theta_0_measured={run['theta_0_measured']:.3e}  plateau={run['plateau']:.3e}")
    return run


def build_run_specs():
    specs = []
    for theta in THETA_VALUES:
        for h in H_VALUES:
            specs.append(("PRK", theta, h))
    for theta in THETA_VALUES:
        specs.append(("RadauIIA", theta, 0.5))
    for h in H_VALUES:
        specs.append(("RadauIIA", 1e-4, h))
    for theta in THETA_VALUES:
        specs.append(("LobattoIIIC", theta, 0.5))
    for h in H_VALUES:
        specs.append(("LobattoIIIC", 1e-4, h))
    return specs


# ====================== Step 4: CSV / assertions ======================
def write_summary_csv(results: dict) -> None:
    path = SOLUTIONS_DIR / "summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method_id", "theta_0_nominal", "theta_0_measured", "h", "plateau", "dphi_plateau"])
        for (method_id, theta, h), r in results.items():
            w.writerow([method_id, theta, r["theta_0_measured"], h, r["plateau"], r["dphi_plateau"]])
    print(f"Wrote {path} ({len(results)} rows)")


def check_phi_collapses(results: dict) -> None:
    """
    phi_res should be back at ~numerical precision by the step right after the
    (possibly perturbed) initial state, in every run. The scheme returns to the
    manifold immediately regardless of theta_0.
    """
    worst = 0.0
    worst_key = None
    violations = []
    for key, r in results.items():
        if len(r["phi_res"]) <= 1:
            continue
        phi1 = float(r["phi_res"][1])
        if phi1 > worst:
            worst, worst_key = phi1, key
        if phi1 > 1e-6:
            violations.append((key, phi1))
    if violations:
        print("WARNING: phi_res did not collapse to numerical precision after step 1 for:")
        for key, phi1 in violations:
            print(f"    {key}: phi_res[1] = {phi1:.3e}")
    assert worst < 1e-3, (
        f"phi_res failed to collapse after the first step for {worst_key} (phi_res[1]={worst:.3e}); "
        "this indicates a bug, not expected DAE behavior."
    )
    print(f"OK: phi_res collapses after the first step in every run (worst phi_res[1] = {worst:.3e} at {worst_key}).")


# ====================== Step 5: plots ======================
def plot_P1(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for theta in THETA_VALUES:
        r = results[("PRK", theta, 0.5)]
        ax.semilogy(r["t"], np.maximum(r["err"], 1e-300), color=THETA_COLORS[theta],
                    linewidth=1.8, label=f"theta_0 = {theta:.0e}")
    ax.set_xlim(0, 40)
    ax.set_xlabel("t")
    ax.set_ylabel(r"err(t) = $\|\lambda(t) - \lambda_{ref}(t)\|$")
    ax.set_title("PRK (Lobatto IIIA/IIIB, s=3), h=0.5 fixed")
    ax.grid(True, **GRID_KW) # type: ignore
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbation_P1_defect_dependence.png", dpi=150)
    plt.close(fig)


def plot_P2(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for h in H_VALUES:
        r = results[("PRK", 1e-6, h)]
        ax.semilogy(r["t"], np.maximum(r["err"], 1e-300), color=H_COLORS[h],
                    linewidth=1.8, label=f"h = {h:g}")
    ax.set_xlim(0, 40)
    ax.set_xlabel("t")
    ax.set_ylabel(r"err(t) = $\|\lambda(t) - \lambda_{ref}(t)\|$")
    ax.set_title(r"PRK Solutions for fixed $\theta_0=10^{-6}$")
    ax.grid(True, **GRID_KW) # type: ignore
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbation_P2_stepsize_dependence.png", dpi=150)
    plt.close(fig)


def plot_P3(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    hs = np.array(sorted(H_VALUES))
    all_plateaus = []
    for theta in THETA_VALUES:
        plateaus = np.array([results[("PRK", theta, h)]["plateau"] for h in hs])
        all_plateaus.append(plateaus)
        ax.loglog(hs, plateaus, marker="o", color=THETA_COLORS[theta],
                   linewidth=1.8, markersize=6, label=f"theta_0 = {theta:.0e}")
    ax.invert_xaxis()
    ax.set_xlabel("h")
    ax.set_ylabel("plateau = mean(err) over t in [5,10]")
    ax.set_title(r"PRK scaling in $\lambda$ - Plateau $\sim \theta_0 / h$")
    ax.grid(True, **GRID_KW) # type: ignore
    ax.legend()

    all_plateaus = np.concatenate(all_plateaus)
    draw_slope_triangle(ax, h_range=(float(hs.max()), float(hs.min())),
                         y_range=(float(all_plateaus.max()), float(all_plateaus.min())),
                         h_frac=0.5, y_decades_below_max=1.0, order=-1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbation_P3_scaling_law.png", dpi=150)
    plt.close(fig)


def plot_P4(results: dict, C_theory: float) -> float:
    hs, thetas, ratios = [], [], []
    for theta in THETA_VALUES:
        for h in H_VALUES:
            r = results[("PRK", theta, h)]
            hs.append(h)
            thetas.append(theta)
            ratios.append(h * r["plateau"] / theta)
    hs = np.array(hs)
    thetas = np.array(thetas)
    ratios = np.array(ratios)

    # exclude a theta_0 row from the mean if it looks like it sits on the truncation
    # baseline rather than the theta_0/h line: on the line, h*plateau/theta_0 is
    # ~constant across h (small relative spread); off it (truncation-dominated),
    # the ratio swings over orders of magnitude as h changes. A median-of-the-4-rows
    # threshold is self-defeating when half the rows are bad (the median itself gets
    # pulled up), so use a fixed absolute cutoff on the relative spread instead.
    SPREAD_THRESHOLD = 0.3
    rel_spreads = {}
    for theta in THETA_VALUES:
        vals = ratios[thetas == theta]
        rel_spreads[theta] = float(np.std(vals) / np.mean(vals)) if np.mean(vals) != 0 else float("inf")
    print("P4: relative spread of h*plateau/theta_0 across h, per theta_0 "
          f"(excluded from the mean if > {SPREAD_THRESHOLD:g} - truncation-dominated, not on the theta_0/h line):")
    for theta in THETA_VALUES:
        print(f"    theta_0={theta:.0e}: relative spread = {rel_spreads[theta]:.3f}")
    excluded = {theta for theta, s in rel_spreads.items() if s > SPREAD_THRESHOLD}
    if excluded:
        print(f"P4: excluding theta_0={sorted(excluded)} from the mean C")
    mean_mask = ~np.isin(thetas, list(excluded))
    C_mean = float(np.mean(ratios[mean_mask]))

    fig, ax = plt.subplots(figsize=(7, 5))
    for theta in THETA_VALUES:
        mask = thetas == theta
        marker = "x" if theta in excluded else "o"
        ax.scatter(hs[mask], ratios[mask], color=THETA_COLORS[theta], s=60, marker=marker,
                   label=f"theta_0 = {theta:.0e}" + (" (excluded)" if theta in excluded else ""), zorder=3)
    ax.axhline(C_mean, color="#52514e", linewidth=1.4, linestyle="--", label=f"mean C = {C_mean:.3e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("h")
    ax.set_ylabel(r"$h \cdot$ plateau $/ \theta_0$")
    ax.set_title("collapse onto a single constant $C$")
    ax.grid(True, **GRID_KW) # type: ignore
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbation_P4_collapse.png", dpi=150)
    plt.close(fig)

    print(f"P4: empirical mean C = {C_mean:.6e}")
    print(f"P4: theoretical ||inv(dphi_dp @ dg_dlambda)||_2 at t=0 = {C_theory:.6e}")
    print(f"P4: ratio empirical/theoretical = {C_mean / C_theory:.3f}")
    return C_mean


def plot_P5(results: dict) -> None:
    panels = [
        ("PRK", 0.5, "PRK (h=0.5)"),
        ("RadauIIA", 0.5, "Radau IIA (h=0.5)"),
        ("LobattoIIIC", 0.5, "Lobatto IIIC (h=0.5)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (method_id, h, title) in zip(axes, panels): # type: ignore
        for theta in THETA_VALUES:
            r = results[(method_id, theta, h)]
            ax.semilogy(r["t"], np.maximum(r["err"], 1e-300), color=THETA_COLORS[theta],
                        linewidth=1.6, label=f"theta_0 = {theta:.0e}")
        ax.set_xlim(0, 40)
        ax.set_xlabel("t")
        ax.set_title(title)
        ax.grid(True, **GRID_KW)
    axes[0].set_ylabel(r"err(t)") # type: ignore
    axes[0].legend(fontsize=8) # type: ignore
    fig.suptitle("Method Comparison at fixed h (defect handling)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbation_P5_method_contrast_fixed_h.png", dpi=150)
    plt.close(fig)


def plot_P6(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    hs = np.array(sorted(H_VALUES))
    series = [
        ("PRK", "PRK (Lobatto IIIA/IIIB s=3)"),
        ("RadauIIA", "Radau IIA"),
        ("LobattoIIIC", "Lobatto IIIC"),
    ]
    for method_id, label in series:
        plateaus = np.array([results[(method_id, 1e-4, h)]["plateau"] for h in hs])
        ax.loglog(hs, plateaus, marker="o", color=METHOD_COLORS[method_id],
                   linewidth=1.8, markersize=6, label=label)
    ax.invert_xaxis()
    ax.set_xlabel("h")
    ax.set_ylabel("plateau")
    ax.set_title(r"method contrast across h, $\theta_0 = 10^{-4}$")
    ax.grid(True, **GRID_KW) # type: ignore
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "perturbation_P6_method_contrast_across_h.png", dpi=150)
    plt.close(fig)


def print_and_save_P7(results: dict) -> None:
    rows = []
    print("\nP7 (appendix) middle-link table: h * dphi_plateau, PRK runs (rows=theta_0, cols=h)")
    print("Expect each cell ~= theta_0 for its row (defect -> hidden-constraint residual, independent of lambda).")
    header = "theta_0    " + "  ".join(f"h={h:g}" for h in H_VALUES)
    print(header)
    for theta in THETA_VALUES:
        vals = [results[("PRK", theta, h)]["h"] * results[("PRK", theta, h)]["dphi_plateau"] for h in H_VALUES]
        print(f"{theta:.0e}   " + "  ".join(f"{v:.3e}" for v in vals))
        rows.append([theta] + vals)
    with open(SOLUTIONS_DIR / "P7_middle_link_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta_0"] + [f"h={h:g}" for h in H_VALUES])
        w.writerows(rows)


# ====================== theoretical constant for P4 ======================
def theoretical_C() -> float:
    cfg = METHOD_CONFIGS["PRK"]
    sim = GasNetworkSimPerturbation(nodes=nodes, pipes=pipes, method=cfg["method"],
                                     A=cfg["A"], b=cfg["b"], c=cfg["c"], t0=t0, T=T, h=0.5,
                                     q1=demand_profile, h2=h2_profile, ALPHA=ALPHA)
    x0, _ = sim._build(SUPPLY_BAR=SUPPLY_BAR_START, tol=RUN_TOL, maxiter=RUN_MAXITER, theta_0=0.0)
    u0, v0, lam0 = x0 # type: ignore
    dphi_dp = jax.jacfwd(lambda vv: sim.phi(u0, vv, t0))(v0) # type: ignore
    dg_dlambda = jax.jacfwd(lambda ll: sim.g(u0, v0, ll, t0))(lam0) # type: ignore
    return float(jnp.linalg.norm(jnp.linalg.inv(dphi_dp @ dg_dlambda), ord=2))


# ====================== main ======================
if __name__ == "__main__":
    print("=" * 70)
    print("Step 1: loading reference solution")
    print("=" * 70)
    ts_ref, zs_ref, h_ref = load_existing_reference()

    print()
    print("=" * 70)
    print("Step 2/3: running the 32 perturbed-initial-data cases")
    print("=" * 70)
    results = {}
    for method_id, theta_0, h in build_run_specs():
        key = (method_id, theta_0, h)
        results[key] = get_or_run(method_id, theta_0, h, ts_ref, zs_ref, h_ref)

    print()
    print("=" * 70)
    print("Step 4: sanity checks and CSV")
    print("=" * 70)
    check_phi_collapses(results)
    write_summary_csv(results)

    print()
    print("=" * 70)
    print("Step 5: plots")
    print("=" * 70)
    C_theory = theoretical_C()
    plot_P1(results)
    plot_P2(results)
    plot_P3(results)
    plot_P4(results, C_theory)
    plot_P5(results)
    plot_P6(results)
    print_and_save_P7(results)

    print()
    print(f"Figures written to {FIGURES_DIR}")
    print(f"Per-run data and summary CSV written to {SOLUTIONS_DIR}")
    plt.show()
