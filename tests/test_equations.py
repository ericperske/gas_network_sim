import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from gasnetwork.topology import Node, Pipe
from gasnetwork.model import E_pip, J_pip, R_pip, B_pip, z_pip, build_node_split, ModelParams
from gasnetwork.basis_functions import R_PiecewiseLinearContinuous
from gasnetwork.equations import f_rk, g_rk, f_prk, g_prk, phi_prk, pressure_to_density


def _build_params(pipes, nodes, alpha=5.9e-3):
    base_r = R_PiecewiseLinearContinuous(pipes=pipes)
    split = build_node_split(nodes)
    B = B_pip(nodes, pipes, basis_function=base_r)
    B_1, B_2 = B.split(split)
    E_fun = E_pip(nodes, pipes, basis_function=base_r).build
    R_fun = R_pip(nodes, pipes, basis_function=base_r).build
    z_fun = z_pip(nodes, pipes, alpha=alpha).build
    J = J_pip(nodes, pipes).build()

    def q1(t):
        return jnp.array([0.3, -0.3])

    def h2(t):
        return split.h_2

    params = ModelParams(J=J, E=E_fun, R=R_fun, z_func=z_fun, B_1=B_1, B_2=B_2, q_1=q1, h_2=h2)
    return params, split


def _heterogeneous_network(refinements):
    nodes = [
        Node(id=1, h=1.0),
        Node(id=2),
        Node(id=3, flux=0.0),
    ]
    pipes = [
        Pipe(id=1, innode=nodes[0], outnode=nodes[1], length=3500.0, flux_refinement=refinements[0]),
        Pipe(id=2, innode=nodes[1], outnode=nodes[2], length=3500.0, flux_refinement=refinements[1]),
    ]
    return nodes, pipes


def test_runge_kutta_functions_shapes_heterogeneous_flux_refinement():
    """Pipes with different flux_refinement values must still produce correctly
    shaped output from every RK/PRK right-hand-side function."""
    nodes, pipes = _heterogeneous_network(refinements=(4, 3))
    params, split = _build_params(pipes, nodes)

    n_rho = sum(pipe.n_elements for pipe in pipes)
    n_q = sum(pipe.n_fluxes for pipe in pipes)

    rho0 = jnp.full(n_rho, pressure_to_density(80.0, 5.9e-3))
    q0 = jnp.zeros(n_q)
    y0 = jnp.concatenate([rho0, q0])
    w0 = jnp.zeros(split.q_1.shape)
    lam0 = jnp.zeros(split.q_1.shape)

    assert f_rk(y0, w0, 0.0, params).shape == (n_rho + n_q,)
    assert g_rk(y0, 0.0, params).shape == split.q_1.shape
    assert f_prk(rho0, q0, 0.0, params).shape == (n_rho,)
    assert g_prk(rho0, q0, lam0, 0.0, params).shape == (n_q,)
    assert phi_prk(rho0, q0, 0.0, params).shape == split.q_1.shape


def test_rk_prk_equivalence_arbitrary_flux_refinement():
    """
    f_prk/g_prk/phi_prk are the rho/q block-split of f_rk/g_rk (E is block
    lower-triangular in [rho, q]), so they must coincide exactly no matter how
    many sub-elements each pipe is refined into.
    """
    np.random.seed(0)
    nodes, pipes = _heterogeneous_network(refinements=(5, 3))
    params, split = _build_params(pipes, nodes)

    n_rho = sum(pipe.n_elements for pipe in pipes)
    n_q = sum(pipe.n_fluxes for pipe in pipes)

    rho0 = jnp.array(np.random.uniform(4.0, 5.0, n_rho))
    q0 = jnp.array(np.random.uniform(-1.0, 1.0, n_q))
    lam0 = jnp.array(np.random.uniform(-1.0, 1.0, split.q_1.shape[0]))
    y0 = jnp.concatenate([rho0, q0])

    f_rk_out = f_rk(y0, lam0, 0.0, params)
    f_prk_out = f_prk(rho0, q0, 0.0, params)
    g_prk_out = g_prk(rho0, q0, lam0, 0.0, params)

    np.testing.assert_allclose(f_rk_out[:n_rho], f_prk_out, atol=1e-10)
    np.testing.assert_allclose(f_rk_out[n_rho:], g_prk_out, atol=1e-10)
    np.testing.assert_allclose(g_rk(y0, 0.0, params), phi_prk(rho0, q0, 0.0, params), atol=1e-10)


def test_rk_prk_equivalence_single_pipe_various_refinements():
    for refinement in (2, 3, 6):
        nodes = [Node(id=1, h=1.0), Node(id=2, flux=0.0)]
        pipes = [Pipe(id=1, innode=nodes[0], outnode=nodes[1], length=2000.0, flux_refinement=refinement)]
        params, split = _build_params(pipes, nodes)

        n_rho = pipes[0].n_elements
        n_q = pipes[0].n_fluxes

        rho0 = jnp.full(n_rho, pressure_to_density(60.0, 5.9e-3))
        q0 = jnp.linspace(-0.5, 0.5, n_q)
        lam0 = jnp.zeros(split.q_1.shape)
        y0 = jnp.concatenate([rho0, q0])

        f_rk_out = f_rk(y0, lam0, 0.0, params)
        f_prk_out = f_prk(rho0, q0, 0.0, params)
        g_prk_out = g_prk(rho0, q0, lam0, 0.0, params)

        np.testing.assert_allclose(
            f_rk_out[:n_rho], f_prk_out, atol=1e-10,
            err_msg=f"f_prk mismatch for flux_refinement={refinement}",
        )
        np.testing.assert_allclose(
            f_rk_out[n_rho:], g_prk_out, atol=1e-10,
            err_msg=f"g_prk mismatch for flux_refinement={refinement}",
        )
