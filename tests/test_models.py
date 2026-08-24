import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from gasnetwork.model import Node, Pipe, E_ma_rho, E_mo_rho, E_ma_q, E_mo_q, B_h_mo, A_ma_q, E_pip, J_pip, R_mo_q, R_pip, B_pip, z_pip

from gasnetwork.basis_functions import P_PiecewiseConstant, R_PiecewiseLinearContinuous
from gasnetwork.constants import GasConstants

def test_E_ma_rho_1():
    node1 = Node(id=1)
    node2 = Node(id=2)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    nodes = [node1, node2]
    pipes = [pipe1]

    # Create the mass matrix for the pressure basis functions
    mass_matrix_rho = E_ma_rho(pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes))
    E_rho = mass_matrix_rho.build()

    # Mass matrix should only contain pipe length
    assert E_rho.shape == (1, 1)
    assert jnp.allclose(E_rho[0, 0], 1000.0)

def test_E_ma_q_1():
    node1 = Node(id=1)
    node2 = Node(id=2)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    nodes = [node1, node2]
    pipes = [pipe1]

    # Create the mass matrix for the flux basis functions
    E_q = E_ma_q(nodes=nodes,pipes=pipes).build()

    expected = jnp.array(
        [
            [0.0, 0.0,],
        ]
    )

    np.testing.assert_array_equal(E_q, jnp.array(expected))

def test_E_mo_q_1():
    node1 = Node(id=1)
    node2 = Node(id=2)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    nodes = [node1, node2]
    pipes = [pipe1]

    E_q = E_mo_q(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build(c=jnp.array([2.0, 3.0, 4.0]))

    expected = (1 / 2) * (1000 / 6) * np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )

    np.testing.assert_array_equal(E_q, expected)

def test_E_mo_rho_1():
    node1 = Node(id=1)
    node2 = Node(id=2)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    nodes = [node1, node2]
    pipes = [pipe1]

    E_rho = E_mo_rho(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build(c=jnp.array([2.0, 3.0, 4.0]))

    expected = (-1) * (1 / 4) * (1000 / 6) * np.array(
        [
            [2 * 3.0 + 4.0],
            [3.0 + 2 * 4.0],
        ]
    )

    np.testing.assert_array_equal(E_rho, expected)

def test_A_ma_q_1():
    node1 = Node(id=1)
    node2 = Node(id=2)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    nodes = [node1, node2]
    pipes = [pipe1]

    A_q = A_ma_q(pipes=pipes, nodes=nodes, basis_function_r=R_PiecewiseLinearContinuous(pipes=pipes), basis_function_p=P_PiecewiseConstant(pipes=pipes)).build()

    expected = np.array(
        [
            [-1.0, 1.0],
        ]
    )

    np.testing.assert_array_equal(A_q, expected)

def test_B_h_mo_1():
    node1 = Node(id=1)
    node2 = Node(id=2)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    nodes = [node1, node2]
    pipes = [pipe1]

    B_h = B_h_mo(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build()

    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )

    np.testing.assert_array_equal(B_h, expected)

def test_E_pip_1():
    node1 = Node(id=1)
    node2 = Node(id=2)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    nodes = [node1, node2]
    pipes = [pipe1]

    E = E_pip(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build(c=jnp.array([2.0, 3.0, 4.0]))

    expected = 1000 * np.array(
        [
            [1.0, 0.0, 0.0],
            [(-1) / (6 * (2.0 ** 2)) * (2 * 3.0 + 4.0), 1 / (3 * 2.0), 1 / (6 * 2.0)],
            [(-1) / (6 * (2.0 ** 2)) * (3.0 + 2 * 4.0), 1 / (6 * 2.0), 1 / (3 * 2.0)],
        ]
    )

    np.testing.assert_array_equal(E, expected)

def test_J_pip_1():
    node1 = Node(id=1)
    node2 = Node(id=2)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    nodes = [node1, node2]
    pipes = [pipe1]

    J = J_pip(nodes=nodes, pipes=pipes).build()

    expected = np.array(
        [
            [0.0, 1.0, -1.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(J, expected)

def test_B_pip_1():
    node1 = Node(id=1)
    node2 = Node(id=2)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    nodes = [node1, node2]
    pipes = [pipe1]

    B_pip_matrix = B_pip(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build()

    expected = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )

    np.testing.assert_array_equal(B_pip_matrix, expected)

def test_R_q_mo_1():
    node1 = Node(id=1)
    node2 = Node(id=2)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    R = R_mo_q(nodes=[node1, node2], pipes=[pipe1], basis_function=R_PiecewiseLinearContinuous(pipes=[pipe1])).build(c=jnp.array([1.0, 2.0, 3.0]))

    expected = pipe1.gamma * 1000.0 / 1.0 * np.array(
        [
            [(3 * 2.0 + 3.0) / 12, (2.0 + 3.0) / 12],
            [(2.0 + 3.0) / 12, (2.0 + 3 * 3.0) / 12],
        ]
    )

    np.testing.assert_array_equal(R, expected)

def test_R_pip_1():
    node1 = Node(id=1)
    node2 = Node(id=2)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    R_mo = R_mo_q(nodes=[node1, node2], pipes=[pipe1], basis_function=R_PiecewiseLinearContinuous(pipes=[pipe1])).build(c=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    R = R_pip(nodes=[node1, node2], pipes=[pipe1], basis_function=R_PiecewiseLinearContinuous(pipes=[pipe1])).build(c=jnp.array([1.0, 2.0, 3.0]))
    expected = pipe1.gamma * 1000.0 / 1.0 * np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, (3 * 2.0 + 3.0) / 12, (2.0 + 3.0) / 12],
            [0.0, (2.0 + 3.0) / 12, (2.0 + 3 * 3.0) / 12],
        ]
    )

    np.testing.assert_array_equal(R, expected)


# =======================================================================================================================================

def test_E_ma_rho_2():
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=500.0)

    nodes = [node1, node2, node3]
    pipes = [pipe1, pipe2]

    # Create the mass matrix for the pressure basis functions
    mass_matrix_rho = E_ma_rho(pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes))
    E_rho = mass_matrix_rho.build()

    # Check that the mass matrix is diagonal and has the correct values
    assert E_rho.shape == (2, 2)
    assert np.allclose(E_rho[0, 0], 1000.0)
    assert np.allclose(E_rho[1, 1], 500.0)
    assert np.allclose(E_rho[0, 1], 0.0)
    assert np.allclose(E_rho[1, 0], 0.0)


def test_E_ma_q_2():
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=500.0)

    nodes = [node1, node2, node3]
    pipes = [pipe1, pipe2]

    E_q = E_ma_q(nodes=nodes,pipes=pipes).build()

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(E_q, expected)

def test_E_mo_rho_2():
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=500.0)

    nodes = [node1, node2, node3]
    pipes = [pipe1, pipe2]

    E_rho = E_mo_rho(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build(c=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    expected = np.array(
        [
            [-1000.0 / (6* (1.0 ** 2))  * (2 * 3.0 + 4.0), 0.0],
            [-1000.0 / (6* (1.0 ** 2))  * (3.0 + 2 * 4.0), 0.0],
            [0.0, -500.0 / (6 * (2.0 ** 2)) * (2 * 5.0 + 6.0)],
            [0.0, -500.0 / (6 * (2.0 ** 2)) * (5.0 + 2 * 6.0)],
        ]
    )

    np.testing.assert_array_equal(E_rho, expected)

def test_E_mo_q_2():
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=500.0)

    nodes = [node1, node2, node3]
    pipes = [pipe1, pipe2]
    #                          c_rh0  |        c_q
    E_q = E_mo_q(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build(c=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    expected = np.array(
        [
            [1000.0 / (3 * 1.0), 1000.0 / (6 * 1.0), 0.0, 0.0],
            [1000.0 / (6 * 1.0), 1000.0 / (3 * 1.0), 0.0, 0.0],
            [0.0, 0.0, 500.0 / (3 * 2.0), 500.0 / (6 * 2.0)],
            [0.0, 0.0, 500.0 / (6 * 2.0), 500.0 / (3 * 2.0)],
        ]
    )

    np.testing.assert_array_equal(E_q, expected)#

def test_A_ma_q_2():
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=500.0)

    nodes = [node1, node2, node3]
    pipes = [pipe1, pipe2]

    A_q = A_ma_q(pipes=pipes, nodes=nodes, basis_function_r=R_PiecewiseLinearContinuous(pipes=pipes), basis_function_p=P_PiecewiseConstant(pipes=pipes)).build()

    expected = np.array(
        [
            [-1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    )

    np.testing.assert_array_equal(A_q, expected)


def test_B_h_mo_2():
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=500.0)

    nodes = [node1, node2, node3]
    pipes = [pipe1, pipe2]

    B_h = B_h_mo(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build()

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0]
        ]
    )

    np.testing.assert_array_equal(B_h, expected)

def test_E_pip_2():
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=500.0)

    E = E_pip(nodes=[node1, node2, node3], pipes=[pipe1, pipe2], basis_function=R_PiecewiseLinearContinuous(pipes=[pipe1, pipe2])).build(c=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    expected = np.array(
        [
            [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 500.0, 0.0, 0.0, 0.0, 0.0],
            [-1000.0 / (6 * (1.0 ** 2)) * (2 * 3.0 + 4.0), 0.0, 1000 / (3 * (1.0 ** 2)), 1000 / (6 * (1.0 ** 2)), 0.0, 0.0],
            [-1000.0 / (6 * (1.0 ** 2)) * (3.0 + 2 * 4.0), 0.0, 1000 / (6 * (1.0 ** 2)), 1000 / (3 * (1.0 ** 2)), 0.0, 0.0],
            [0.0, -500.0 / (6 * (2.0 ** 2)) * (2 * 5.0 + 6.0), 0.0, 0.0, 1000 / (3 * (2.0 ** 2)), 1000 / (6 * (2.0 ** 2))],
            [0.0, -500.0 / (6 * (2.0 ** 2)) * (5.0 + 2 * 6.0), 0.0, 0.0, 1000 / (6 * (2.0 ** 2)), 1000 / (3 * (2.0 ** 2))],
        ]
    )

    np.testing.assert_array_equal(E, expected)


def test_J_pip_2():
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=500.0)

    nodes = [node1, node2, node3]
    pipes = [pipe1, pipe2]

    J = J_pip(nodes=nodes, pipes=pipes).build()

    expected = np.array(
        [
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(J, expected)

def test_B_pip_2():
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=500.0)

    nodes = [node1, node2, node3]
    pipes = [pipe1, pipe2]

    B_pip_matrix = B_pip(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build()

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    np.testing.assert_array_equal(B_pip_matrix, expected)

def test_R_q_mo_2():
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=500.0)

    nodes = [node1, node2, node3]
    pipes = [pipe1, pipe2]

    R = R_mo_q(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build(c=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    expected = np.array(
        [
            [pipe1.gamma / 1.0 * 1000.0/12 * (3*3.0 + 4.0), pipe1.gamma / 1.0 * 1000.0/12 * (3.0 + 4.0),     0.0, 0.0],
            [pipe1.gamma / 1.0 * 1000.0/12 * (3.0 + 4.0),   pipe1.gamma / 1.0 * 1000.0/12 * (3.0 + 3*4.0),   0.0, 0.0],
            [0.0, 0.0, pipe2.gamma / 2.0 * 500.0/12 * (3*5.0 + 6.0), pipe2.gamma / 2.0 * 500.0/12 * (5.0 + 6.0)],
            [0.0, 0.0, pipe2.gamma / 2.0 * 500.0/12 * (5.0 + 6.0),   pipe2.gamma / 2.0 * 500.0/12 * (5.0 + 3*6.0)],
        ]
    )
    np.testing.assert_allclose(R, expected, atol=1e-10) # the differences were of order 10^-16, so numerically negligable

def test_R_pip_2():
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=500.0)

    nodes = [node1, node2, node3]
    pipes = [pipe1, pipe2]

    R = R_pip(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build(c=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, pipe1.gamma / 1.0 * 1000.0/12 * (3*3.0 + 4.0), pipe1.gamma / 1.0 * 1000.0/12 * (3.0 + 4.0),     0.0, 0.0],
            [0.0, 0.0, pipe1.gamma / 1.0 * 1000.0/12 * (3.0 + 4.0), pipe1.gamma / 1.0 * 1000.0/12 * (3.0 + 3*4.0),   0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, pipe2.gamma / 2.0 * 500.0/12 * (3*5.0 + 6.0), pipe2.gamma / 2.0 * 500.0/12 * (5.0 + 6.0)],
            [0.0, 0.0, 0.0, 0.0, pipe2.gamma / 2.0 * 500.0/12 * (5.0 + 6.0),   pipe2.gamma / 2.0 * 500.0/12 * (5.0 + 3*6.0)],
        ]
    )
    np.testing.assert_allclose(R, expected, atol=1e-10) # c.f. test R_mo_q


# =======================================================================================================================================
# flux_refinement: a single pipe refined into multiple sub-elements (flux_refinement=3 -> 2 elements)
# =======================================================================================================================================

def _refined_pipe_setup():
    node1 = Node(id=1)
    node2 = Node(id=2)
    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0, flux_refinement=3)
    nodes = [node1, node2]
    pipes = [pipe1]
    return node1, node2, pipe1, nodes, pipes


def test_E_ma_rho_flux_refinement():
    _, _, pipe1, _, pipes = _refined_pipe_setup()
    E_rho = E_ma_rho(pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build()

    assert E_rho.shape == (2, 2)
    np.testing.assert_allclose(E_rho, np.diag([500.0, 500.0]))


def test_A_ma_q_flux_refinement():
    _, _, pipe1, nodes, pipes = _refined_pipe_setup()
    A_q = A_ma_q(pipes=pipes, nodes=nodes, basis_function_r=R_PiecewiseLinearContinuous(pipes=pipes), basis_function_p=P_PiecewiseConstant(pipes=pipes)).build()

    expected = np.array(
        [
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(A_q, expected)


def test_E_mo_rho_flux_refinement():
    _, _, pipe1, nodes, pipes = _refined_pipe_setup()
    c = jnp.array([2.0, 3.0, 1.0, 2.0, 3.0])  # c_rho = [2.0, 3.0], c_q = [1.0, 2.0, 3.0]
    E_rho = E_mo_rho(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build(c)

    h = 500.0
    expected = np.array(
        [
            [-(1.0 * (h / 3) + 2.0 * (h / 6)) / 2.0 ** 2, 0.0],
            [-(1.0 * (h / 6) + 2.0 * (h / 3)) / 2.0 ** 2, -(2.0 * (h / 3) + 3.0 * (h / 6)) / 3.0 ** 2],
            [0.0, -(2.0 * (h / 6) + 3.0 * (h / 3)) / 3.0 ** 2],
        ]
    )
    np.testing.assert_allclose(E_rho, expected, atol=1e-10)


def test_E_mo_q_flux_refinement_accumulates_at_shared_node():
    _, _, pipe1, nodes, pipes = _refined_pipe_setup()
    c = jnp.array([2.0, 3.0, 1.0, 2.0, 3.0])
    E_q = E_mo_q(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build(c)

    h = 500.0
    rho0, rho1 = 2.0, 3.0
    expected = np.zeros((3, 3))
    expected[0, 0] += (h / 3) / rho0
    expected[0, 1] += (h / 6) / rho0
    expected[1, 0] += (h / 6) / rho0
    expected[1, 1] += (h / 3) / rho0
    expected[1, 1] += (h / 3) / rho1
    expected[1, 2] += (h / 6) / rho1
    expected[2, 1] += (h / 6) / rho1
    expected[2, 2] += (h / 3) / rho1

    np.testing.assert_allclose(E_q, expected, atol=1e-10)
    # the shared interior flux coefficient (index 1) must accumulate contributions from
    # BOTH neighboring elements, not just overwrite one with the other
    assert expected[1, 1] == (h / 3) / rho0 + (h / 3) / rho1


def test_R_mo_q_flux_refinement_accumulates_at_shared_node():
    _, _, pipe1, nodes, pipes = _refined_pipe_setup()
    c = jnp.array([2.0, 3.0, 1.0, 2.0, 3.0])
    R = R_mo_q(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build(c)

    h = 500.0
    gamma = pipe1.gamma

    def local_block(rho, q_in, q_out):
        # closed form for q_in, q_out same sign (both positive here), c.f. test_R_q_mo_1/2
        return (gamma / rho) * (h / 12) * np.array(
            [
                [3 * q_in + q_out, q_in + q_out],
                [q_in + q_out, q_in + 3 * q_out],
            ]
        )

    R0 = local_block(2.0, 1.0, 2.0)  # element 0: rho=2.0, q_in=1.0, q_out=2.0
    R1 = local_block(3.0, 2.0, 3.0)  # element 1: rho=3.0, q_in=2.0, q_out=3.0

    expected = np.zeros((3, 3))
    expected[0:2, 0:2] += R0
    expected[1:3, 1:3] += R1

    np.testing.assert_allclose(R, expected, atol=1e-8)


def test_z_pip_flux_refinement():
    _, _, pipe1, nodes, pipes = _refined_pipe_setup()
    alpha = 1e-4
    c = jnp.array([2.0, 3.0, 1.0, 2.0, 3.0])
    z = z_pip(nodes=nodes, pipes=pipes, alpha=alpha).build(c)

    RT = (GasConstants.R / GasConstants.M) / GasConstants.PA_PER_BAR * GasConstants.T

    def z_h_local(rho, q_in, q_out):
        kinetic = (q_in ** 2 + q_in * q_out + q_out ** 2) / (6.0 * rho ** 2)
        pressure = RT * np.log((rho * (1 - alpha * RT)) / (1 - alpha * RT * rho)) + RT / (1 - alpha * RT * rho)
        return kinetic + pressure  # pipe.angle defaults to 0.0, so slope term is 0

    expected_zh = np.array([z_h_local(2.0, 1.0, 2.0), z_h_local(3.0, 2.0, 3.0)])
    expected = np.concatenate([expected_zh, np.array([1.0, 2.0, 3.0])])

    assert z.shape == (5,)
    np.testing.assert_allclose(z, expected, atol=1e-10)


def test_B_h_mo_flux_refinement():
    _, _, pipe1, nodes, pipes = _refined_pipe_setup()
    B_h = B_h_mo(nodes=nodes, pipes=pipes, basis_function=R_PiecewiseLinearContinuous(pipes=pipes)).build()

    expected = np.array(
        [
            [1.0, 0.0],   # local flux coefficient 0: real innode boundary
            [0.0, 0.0],   # local flux coefficient 1: interior coefficient, no boundary coupling
            [0.0, -1.0],  # local flux coefficient 2: real outnode boundary
        ]
    )
    np.testing.assert_array_equal(B_h, expected)


def test_heterogeneous_flux_refinement_shapes():
    """Two pipes with different flux_refinement values must assemble consistently,
    with only each pipe's own two boundary flux coefficients coupling to real graph nodes."""
    node1 = Node(id=1)
    node2 = Node(id=2)
    node3 = Node(id=3)

    pipe1 = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)                # flux_refinement=2 (default)
    pipe2 = Pipe(id=2, innode=node2, outnode=node3, length=600.0, flux_refinement=4)

    nodes = [node1, node2, node3]
    pipes = [pipe1, pipe2]

    n_rho_expected = pipe1.n_elements + pipe2.n_elements  # 1 + 3
    n_q_expected = pipe1.n_fluxes + pipe2.n_fluxes          # 2 + 4

    basis = R_PiecewiseLinearContinuous(pipes=pipes)
    c = jnp.concatenate([jnp.ones(n_rho_expected), jnp.arange(1.0, n_q_expected + 1)])

    E = E_pip(nodes=nodes, pipes=pipes, basis_function=basis).build(c=c)
    assert E.shape == (n_rho_expected + n_q_expected, n_rho_expected + n_q_expected)

    J = J_pip(nodes=nodes, pipes=pipes).build()
    assert J.shape == (n_rho_expected + n_q_expected, n_rho_expected + n_q_expected)

    B = B_pip(nodes=nodes, pipes=pipes, basis_function=basis).build()
    assert B.shape == (n_rho_expected + n_q_expected, len(nodes))

    # pipe2's interior flux rows (local indices 1, 2 out of n_elements=3) must be
    # all-zero: they are private to the pipe, not real junctions. B stacks all
    # n_rho_expected rho-rows before the q-block, so pipe2's q-block starts at
    # n_rho_expected + pipe1.n_fluxes.
    q2_offset = n_rho_expected + pipe1.n_fluxes
    interior_rows = B[q2_offset + 1 : q2_offset + pipe2.n_elements, :]
    np.testing.assert_array_equal(interior_rows, np.zeros_like(interior_rows))


if __name__ == "__main__":
    test_A_ma_q_1()
    test_E_ma_rho_1()
    test_E_ma_q_1()
    test_E_mo_rho_1()
    test_E_mo_q_1()
    test_E_pip_1()
    test_J_pip_1()
    test_R_q_mo_1()
    test_R_pip_1()
    test_B_h_mo_1()
    test_B_pip_1()
    print("All tests for single pipe network passed!")

    test_A_ma_q_2()
    test_E_ma_rho_2()
    test_E_ma_q_2()
    test_E_mo_rho_2()
    test_E_mo_q_2()    
    test_E_pip_2()
    test_J_pip_2()
    test_R_q_mo_2()
    test_R_pip_2()
    test_B_h_mo_2()
    test_B_pip_2()
    print("All tests for the two pipe network passed!")

    print(100*"=")
    print("All tests passed successfully!")