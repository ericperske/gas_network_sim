from typing import Sequence, Union, Tuple, Literal, Callable

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from gasnetwork.topology import Node, Pipe
from gasnetwork.solver import GasNetworkSim

# ====================== define downsampling function for plotting ======================
def downsample_by_time(t: jnp.ndarray, *arrays: jnp.ndarray, dt: float = 0.1) -> tuple:
    """
    Downsamples t and any number of arrays (aligned along their first axis) so that
    consecutive samples are spaced at least dt apart, assuming t is uniformly spaced.
    Keeps plots of finely-resolved solutions (small step size h) readable instead of
    bloating them with one point per solver step.
    """
    t = jnp.asarray(t)
    step = max(1, int(round(dt / float(t[1] - t[0]))))
    return (t[::step],) + tuple(a[::step] for a in arrays)

# ====================== define write-results-function ======================
def save_ref_solution(name: str, t: jnp.ndarray, rho: jnp.ndarray, q: jnp.ndarray, z: jnp.ndarray, h: float):
    path =  Path(__file__).resolve().parent / f"{name}.npz"
    np.savez_compressed(
        path, 
        t=np.asarray(t),
        rho=np.asarray(rho),
        q=np.asarray(q),
        z=np.asarray(z),
        h_ref=h,
    )

# ====================== define read-results-function ======================
def load_reference_solution(path: Path, as_jax: bool = True) -> dict:
    npz = np.load(path, allow_pickle=False)
    data = {k: npz[k] for k in npz.files}
    if as_jax:
        for key in ("t", "rho", "q", "z"):
            data[key] = jnp.asarray(data[key])
    return data

# ====================== define plotting function for experiments ======================
def  plot_convergence(results: dict[str, dict[str, tuple[jnp.ndarray, jnp.ndarray]]], combined: bool = True,
                      xlabel: str = "Step size h", ylabel: str = "Error", title: str = "Convergence"
                      ) -> Union[Axes, jnp.ndarray]:
    """
    Log-log convergence plot(s). `results` maps a variable name (e.g. "y", "w" or
    "rho", "q", "lambda") to a dict of {series_label: (h_values, error_values)}
    (e.g. one series per stage count s).

    If combined is True (default), every variable/series pair is drawn on a single
    shared Axes. If False, each variable gets its own separate, full-size figure
    (rather than a subplot squeezed alongside the others), so each one reads
    clearly and can be saved on its own.
    """
    if combined:
        _, ax = plt.subplots()
        for var_name, series in results.items():
            for label, (h, err) in series.items():
                ax.loglog(h, err, marker="o", label=f"{var_name}: {label}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.invert_xaxis()  # numerics convention: coarse (large h) on the left, refining rightward
        ax.legend()
        return ax

    axes = []
    for var_name, series in results.items():
        _, ax = plt.subplots()
        for label, (h, err) in series.items():
            ax.loglog(h, err, marker="o", label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} ({var_name})")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.invert_xaxis()  # numerics convention: coarse (large h) on the left, refining rightward
        ax.legend()
        axes.append(ax)
    return np.array(axes) # type: ignore


def draw_slope_triangle(ax: Axes, h_range: Tuple[float, float], y_range: Tuple[float, float],
                         h_frac: float, y_decades_below_max: float, order: float,
                         width_decades: float = 0.4, color: str = "red",
                         linewidth: float = 1.2, fontsize: float = 9) -> None:
    """
    Draw a single triangle whose hypotenuse has the given log-log slope
    (order), purely as a visual reference:
    """
    h_max, h_min = h_range
    y_max, _ = y_range
    h_anchor = 10 ** (jnp.log10(h_max) - h_frac * (jnp.log10(h_max) - jnp.log10(h_min)))
    y_anchor = y_max * 10 ** (-y_decades_below_max)

    h2 = h_anchor * 10 ** (-width_decades)
    y2 = y_anchor * (h2 / h_anchor) ** order

    ax.plot([h_anchor, h2, h2, h_anchor], [y_anchor, y_anchor, y2, y_anchor], # type: ignore
            color=color, linewidth=linewidth)

    h_label = float(jnp.sqrt(h_anchor * h2))
    y_label = float(jnp.sqrt(y_anchor * y2))
    ax.text(h_label, y_label * 1.15, f"{order:g}", color=color, fontsize=fontsize,
            ha="center", va="bottom")


def run_convergence_study_network(nodes: Sequence[Node], pipes: Sequence[Pipe],
                           method: Literal["RungeKutta", "PartitionedRungeKutta"],
                           A: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], b: jnp.ndarray, c: jnp.ndarray,
                           h_values: Sequence[float], reference: dict, t0: float, T: float, q1: Callable, h2: Callable,
                           SUPPLY_BAR: float, ALPHA: float, tol: float = 1e-6, maxiter: int = 1000
                           ) -> dict[str, tuple[jnp.ndarray, jnp.ndarray]]:
    ref_idx = int(T / reference["h_ref"])
    rho_ref = jnp.asarray(reference["rho"])[ref_idx]
    q_ref = jnp.asarray(reference["q"])[ref_idx]
    z_ref = jnp.asarray(reference["z"])[ref_idx]
    y_ref = jnp.concatenate([rho_ref, q_ref])

    if method == "RungeKutta":
        errors = {"y": [], "w": []}
    else:
        errors = {"rho": [], "q": [], "lambda": []}

    for h in h_values:
        sim = GasNetworkSim(nodes=nodes, pipes=pipes, method=method, A=A, b=b, c=c,
                             t0=t0, T=T, h=h, q1=q1, h2=h2, ALPHA=ALPHA)
        result = sim.simulate(SUPPLY_BAR=SUPPLY_BAR, tol=tol, maxiter=maxiter)  # type: ignore
        print("Current stepsize: ", h)
        if method == "RungeKutta":
            y_final, z_final = result[1][-1], result[2][-1]
            errors["y"].append(jnp.sqrt(jnp.sum((y_final - y_ref) ** 2)))
            errors["w"].append(jnp.sqrt(jnp.sum((z_final - z_ref) ** 2)))
        else:
            rho_final, q_final, lambda_final = result[1][-1], result[2][-1], result[3][-1]  # type: ignore
            errors["rho"].append(jnp.sqrt(jnp.sum((rho_final - rho_ref) ** 2)))
            errors["q"].append(jnp.sqrt(jnp.sum((q_final - q_ref) ** 2)))
            errors["lambda"].append(jnp.sqrt(jnp.sum((lambda_final - z_ref) ** 2)))

    h_array = jnp.asarray(h_values, dtype=float)
    return {var_name: (h_array, jnp.asarray(err)) for var_name, err in errors.items()}


def empirical_order(h: jnp.ndarray, err: jnp.ndarray) -> jnp.ndarray:
    """
    Observed convergence order between consecutive (h, error) points:
    order[i] = log(err[i]/err[i+1]) / log(h[i]/h[i+1]).
    Only meaningful while err is still in the truncation-error regime;
    it becomes unreliable once err flattens out at the round-off floor.
    """
    h = jnp.asarray(h, dtype=float)
    err = jnp.asarray(err, dtype=float)
    return jnp.log(err[:-1] / err[1:]) / jnp.log(h[:-1] / h[1:])
