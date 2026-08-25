import jax
import jax.numpy as jnp
from typing import Sequence, Callable
from dataclasses import dataclass

from gasnetwork.basis_functions import P_PiecewiseConstant, R_PiecewiseLinearContinuous
from gasnetwork.topology import Node, Pipe
from gasnetwork.constants import GasConstants

# ========================= coefficient layout helpers =========================
def _rho_offsets(pipes: Sequence[Pipe]) -> tuple[list[int], int]:
    """Per-pipe start index into the global density (rho) coefficient vector, plus the total count."""
    offsets = []
    offset = 0
    for pipe in pipes:
        offsets.append(offset)
        offset += pipe.n_elements
    return offsets, offset


def _q_offsets(pipes: Sequence[Pipe]) -> tuple[list[int], int]:
    """Per-pipe start index into the global flux (q) coefficient vector, plus the total count."""
    offsets = []
    offset = 0
    for pipe in pipes:
        offsets.append(offset)
        offset += pipe.n_fluxes
    return offsets, offset


# ========================= model utility functions =========================
@dataclass
class NodeSplit:
    given_q_idx: jnp.ndarray    # flux given, h unknown -> B1
    given_h_idx: jnp.ndarray    # enthalpy given, flux is output -> B2
    q_1: jnp.ndarray             # given flux values, aligned with given_q_idx
    h_2: jnp.ndarray             # given enthalpy values, aligned with given_h_idx


def build_node_split(nodes: Sequence[Node]) -> NodeSplit:
    given_q_idx = [i for i, n in enumerate(nodes) if n.is_flux_node or n.is_interior]
    given_h_idx = [i for i, n in enumerate(nodes) if n.is_enthalpy_node]

    if not given_q_idx:
        raise ValueError("No flux-prescribed nodes: B_1 would be empty.")

    q_1 = jnp.array([
        nodes[i].flux if nodes[i].is_flux_node else 0.0
        for i in given_q_idx
    ], dtype=jnp.float64)

    h_2 = jnp.array([
        nodes[i].enthalpy 
        for i in given_h_idx
    ], dtype=jnp.float64)

    return NodeSplit(
        given_q_idx=jnp.array(given_q_idx, dtype=int),
        given_h_idx=jnp.array(given_h_idx, dtype=int),
        q_1=q_1,
        h_2=h_2,
    )

def classify_nodes(nodes: Sequence[Node]) -> dict[str, list[int]]:
    """
    Classify nodes into interior, enthalpy boundary, and flux boundary nodes.
    """
    given_h = [i for i, node in enumerate(nodes) if node.is_enthalpy_node]
    given_q = [i for i, node in enumerate(nodes) if node.is_flux_node]
    interior = [i for i, node in enumerate(nodes) if node.is_interior]
    return {
        "given_h": given_h,
        "given_q": given_q,
        "interior": interior,
    }


# ========================= param data class =========================
@dataclass
class ModelParams:
    J: jnp.ndarray
    E: Callable[[jnp.ndarray], jnp.ndarray]
    R: Callable[[jnp.ndarray], jnp.ndarray]
    z_func: Callable[[jnp.ndarray], jnp.ndarray]
    B_1: jnp.ndarray
    B_2: jnp.ndarray
    q_1: Callable[[float], jnp.ndarray]
    h_2: Callable[[float], jnp.ndarray]
    
# ========================= model matrices =========================
class A_ma_q:
    """
    Mass balance matrix for the pressure and flux basis functions.
    """
    def __init__(self, pipes: Sequence[Pipe], nodes: Sequence[Node],basis_function_r: R_PiecewiseLinearContinuous, basis_function_p: P_PiecewiseConstant):
        self.pipes = pipes
        self.nodes = nodes
        self.basis_function_r = basis_function_r
        self.basis_function_p = basis_function_p

    def build(self) -> jnp.ndarray:
        rho_offsets, n_rho = _rho_offsets(self.pipes)
        q_offsets, n_q = _q_offsets(self.pipes)
        A = jnp.zeros((n_rho, n_q))

        for e, pipe in enumerate(self.pipes):
            for k in range(pipe.n_elements):
                row = rho_offsets[e] + k
                A = A.at[row, q_offsets[e] + k].set(-1.0)
                A = A.at[row, q_offsets[e] + k + 1].set(1.0)

        return A

class E_ma_rho:
    """
    Mass matrix for the pressure basis functions.
    """
    def __init__(self, pipes: Sequence[Pipe], basis_function: R_PiecewiseLinearContinuous):
        self.pipes = pipes
        self.basis_function = basis_function

    def build(self) -> jnp.ndarray:
        rho_offsets, n_rho = _rho_offsets(self.pipes)
        diag_vals = jnp.zeros(n_rho)

        for e, pipe in enumerate(self.pipes):
            for k in range(pipe.n_elements):
                diag_vals = diag_vals.at[rho_offsets[e] + k].set(pipe.element_length)

        return jnp.diag(diag_vals)

class E_ma_q:
    """
    Placeholder for the zero block within E_pip.
    """
    def __init__(self, pipes: Sequence[Pipe], nodes: Sequence[Node]):
        self.pipes = pipes
        self.nodes = nodes
    def build(self) -> jnp.ndarray:
        return jnp.zeros((len(self.pipes), len(self.nodes)))

class E_mo_rho:
    """
    Momentum balance matrix for the pressure basis functions.
    """
    def __init__(self, nodes: Sequence[Node],pipes: Sequence[Pipe], basis_function: R_PiecewiseLinearContinuous):
        self.nodes = nodes
        self.pipes = pipes
        self.basis_function = basis_function
    
    def build(self, c: jnp.ndarray, ) -> jnp.ndarray:
        rho_offsets, n_rho = _rho_offsets(self.pipes)
        q_offsets, n_q = _q_offsets(self.pipes)
        c_rho = c[:n_rho]
        c_q = c[n_rho:]

        E_mo = jnp.zeros((n_q, n_rho))

        for e, pipe in enumerate(self.pipes):
            h = pipe.element_length
            for k in range(pipe.n_elements):
                rho_ek = c_rho[rho_offsets[e] + k]
                q_in = c_q[q_offsets[e] + k]      # flux coefficient at the element's left node
                q_out = c_q[q_offsets[e] + k + 1] # flux coefficient at the element's right node
                col = rho_offsets[e] + k

                E_mo = E_mo.at[q_offsets[e] + k,     col].set(-(q_in * (h / 3) + q_out * (h / 6)) / rho_ek ** 2)
                E_mo = E_mo.at[q_offsets[e] + k + 1, col].set(-(q_in * (h / 6) + q_out * (h / 3)) / rho_ek ** 2)

        return E_mo
    
class E_mo_q:
    """
    Momentum balance matrix for the flux basis functions.
    """
    def __init__(self, nodes: Sequence[Node],pipes: Sequence[Pipe], basis_function: R_PiecewiseLinearContinuous):
        self.nodes = nodes
        self.pipes = pipes
        self.basis_function = basis_function

    def build(self, c: jnp.ndarray, ) -> jnp.ndarray:
        rho_offsets, n_rho = _rho_offsets(self.pipes)
        q_offsets, n_q = _q_offsets(self.pipes)
        c_rho = c[:n_rho]

        E_mo = jnp.zeros((n_q, n_q))

        for e, pipe in enumerate(self.pipes):
            h = pipe.element_length
            for k in range(pipe.n_elements):
                rho_ek = c_rho[rho_offsets[e] + k]
                i = q_offsets[e] + k  # local block start
                # adjacent elements share the flux coefficient at their common interior node, so diagonal contributions must accumulate, do not overwrite!
                E_mo = E_mo.at[i,     i    ].add((h / 3) / rho_ek)
                E_mo = E_mo.at[i,     i + 1].add((h / 6) / rho_ek)
                E_mo = E_mo.at[i + 1, i    ].add((h / 6) / rho_ek)
                E_mo = E_mo.at[i + 1, i + 1].add((h / 3) / rho_ek)

        return E_mo

class B_h_mo:
    """
    Boundary input matrix for the momentum equations.
    B_h^mo[i, k] = r_i(nu_k) at innode, -r_i(nu_k) at outnode.
    """
    def __init__(self, nodes: Sequence[Node], pipes: Sequence[Pipe], basis_function: R_PiecewiseLinearContinuous):
        self.nodes = nodes
        self.pipes = pipes
        self.basis_function = basis_function

    def build(self) -> jnp.ndarray:
        q_offsets, n_q = _q_offsets(self.pipes)
        V = len(self.nodes)
        B_h = jnp.zeros((n_q, V))
        node_to_index = {node: idx for idx, node in enumerate(self.nodes)}

        for e, pipe in enumerate(self.pipes):
            k = node_to_index[pipe.innode]
            l = node_to_index[pipe.outnode]
            # only the pipe's two boundary flux coefficients (local index 0 and
            # n_elements) couple to real graph nodes; any interior flux coefficients
            # from refinement are private to the pipe and get no boundary row
            first_row = q_offsets[e]
            last_row = q_offsets[e] + pipe.n_elements

            B_h = B_h.at[first_row, k].set(+self.basis_function.evaluate(0.0, 0, e))
            B_h = B_h.at[first_row, l].set(-self.basis_function.evaluate(pipe.length, 0, e))
            B_h = B_h.at[last_row, k].set(+self.basis_function.evaluate(0.0, pipe.n_elements, e))
            B_h = B_h.at[last_row, l].set(-self.basis_function.evaluate(pipe.length, pipe.n_elements, e))

        return B_h

class B_pip:
    """
    Full input matrix with a zero block for the pressure equations and a B_h^mo block for the momentum equations.
    """
    def __init__(self, nodes: Sequence[Node], pipes: Sequence[Pipe], basis_function: R_PiecewiseLinearContinuous):
        self.nodes = nodes
        self.pipes = pipes
        self.basis_function = basis_function

    def build(self) -> jnp.ndarray:
        _, n_rho = _rho_offsets(self.pipes)
        _, n_q = _q_offsets(self.pipes)
        V = len(self.nodes)
        B = jnp.zeros((n_rho + n_q, V))
        B_h = B_h_mo(nodes=self.nodes, pipes=self.pipes, basis_function=self.basis_function).build()
        B = B.at[n_rho:, :].set(B_h)
        return B
    
    def split(self, split: NodeSplit) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Split the input matrix into two submatrices corresponding to the given flux and enthalpy nodes.
        """
        B = self.build()
        B1 = B[:, split.given_q_idx]
        B2 = B[:, split.given_h_idx]
        return B1, B2

class z_pip:
    """
    L^2 orthogonal projection onto P_h.
    """
    def __init__(self, nodes: Sequence[Node], pipes: Sequence[Pipe], alpha, consts=GasConstants):
        self.nodes = nodes
        self.pipes = pipes
        # J/(kg*K) converted to bar/(kg/m^3 * K), so P(rho) below comes out in bar
        # (rho is a mass density, kg/m^3) instead of Pa.
        self.R_specific = (consts.R / consts.M) / consts.PA_PER_BAR
        self.T = consts.T
        self.g = consts.g
        self.alpha = alpha

    def _P_prime(self, rho: jax.Array) -> jax.Array:
        """
        P'(rho) for P(rho) = R_specific * T * rho / (1 - alpha * R_specific * T * rho),
        with P(rho) in bar.
        """
        RT = self.R_specific * self.T
        out = RT * jnp.log((rho * (1 - self.alpha * RT)) / (1 - self.alpha * RT * rho)).astype(float) + RT / (1 - self.alpha * RT * rho)
        return out

    def build(self, c: jnp.ndarray) -> jnp.ndarray:
        rho_offsets, n_rho = _rho_offsets(self.pipes)
        q_offsets, n_q = _q_offsets(self.pipes)
        c_rho = c[:n_rho]
        c_q = c[n_rho:]
        z_h = jnp.zeros(n_rho)

        for e, pipe in enumerate(self.pipes):
            slope = self.g * jnp.sin(pipe.angle)
            for k in range(pipe.n_elements):
                rho_ek = c_rho[rho_offsets[e] + k]
                q_in, q_out = c_q[q_offsets[e] + k], c_q[q_offsets[e] + k + 1]

                kinetic = (q_in ** 2 + q_in * q_out + q_out ** 2) / (6.0 * rho_ek ** 2)
                pressure = self._P_prime(rho_ek)

                z_h = z_h.at[rho_offsets[e] + k].set(kinetic + pressure + slope)

        return jnp.concatenate([z_h, c_q])  # first n_rho entries are zero for the pressure coefficients

class E_pip:
    def __init__(self, nodes: Sequence[Node], pipes: Sequence[Pipe], basis_function: R_PiecewiseLinearContinuous):
        self.nodes = nodes
        self.pipes = pipes
        self.basis_function = basis_function

        # static sub-blocks that don't depend on c — build once
        self._E_rho_ma = E_ma_rho(pipes, basis_function).build()
        self._E_q_ma = E_ma_q(pipes, nodes).build()
        self._E_rho_mo_builder = E_mo_rho(nodes, pipes, basis_function)
        self._E_q_mo_builder = E_mo_q(nodes, pipes, basis_function)

    def build(self, c: jnp.ndarray) -> jnp.ndarray:
        _, n_rho = _rho_offsets(self.pipes)
        _, n_q = _q_offsets(self.pipes)
        E_rho_mo = self._E_rho_mo_builder.build(c)
        E_q_mo = self._E_q_mo_builder.build(c)

        E = jnp.zeros((n_rho + n_q, n_rho + n_q))
        E = E.at[:n_rho, :n_rho].set(self._E_rho_ma)
        E = E.at[n_rho:, :n_rho].set(E_rho_mo)
        E = E.at[n_rho:, n_rho:].set(E_q_mo)
        return E


class J_pip:
    def __init__(self, nodes: Sequence[Node], pipes: Sequence[Pipe]):
        self.nodes = nodes
        self.pipes = pipes

    def build(self) -> jnp.ndarray:
        _, n_rho = _rho_offsets(self.pipes)
        _, n_q = _q_offsets(self.pipes)
        J = jnp.zeros((n_rho + n_q, n_rho + n_q))

        A = A_ma_q(nodes=self.nodes, pipes=self.pipes, basis_function_r=R_PiecewiseLinearContinuous(pipes=self.pipes), basis_function_p=P_PiecewiseConstant(pipes=self.pipes)).build()  # shape (n_rho, n_q)

        J = J.at[:n_rho, n_rho:].set(-A)       # upper-right: -A_q^ma
        J = J.at[n_rho:, :n_rho].set(A.T)      # lower-left:  (A_q^ma)^T

        return J
    
class R_mo_q:
    def __init__(self, nodes: Sequence[Node], pipes: Sequence[Pipe], basis_function: R_PiecewiseLinearContinuous):
        self.nodes = nodes
        self.pipes = pipes
        self.basis_function = basis_function

    def _integral_v_ri_rj(self, q_in, q_out, h: float, i: int, j: int, a, b):
        """
        Definite integral of v(x)*r_i(x)*r_j(x) on [a, b] (local, element-length-h
        coordinates) via Simpson's rule, which is exact here since v, r_i and r_j are
        each affine, making the product a cubic.
        """
        r = (lambda x: 1 - x / h, lambda x: x / h)

        def integrand(x):
            v = q_in * r[0](x) + q_out * r[1](x)
            return v * r[i](x) * r[j](x)

        mid = 0.5 * (a + b)
        return (b - a) / 6.0 * (integrand(a) + 4.0 * integrand(mid) + integrand(b))

    def _integrate_abs_v_ri_rj(self, q_in, q_out, h: float, i: int, j: int):
        sign = jnp.where(q_in >= 0, 1.0, -1.0)

        same_sign_integral = sign * self._integral_v_ri_rj(q_in, q_out, h, i, j, 0.0, h)

        # x_star is where v(x) changes sign. Guard the denominator so the split branch
        # (unused whenever q_in, q_out share a sign) never divides 0/0 into a NaN that
        # would otherwise leak into gradients through jnp.where.
        denom = jnp.where(q_in == q_out, 1.0, q_in - q_out)
        x_star = h * q_in / denom
        split_integral = (
            sign * self._integral_v_ri_rj(q_in, q_out, h, i, j, 0.0, x_star)
            - sign * self._integral_v_ri_rj(q_in, q_out, h, i, j, x_star, h)
        )

        return jnp.where(q_in * q_out >= 0, same_sign_integral, split_integral)

    def build(self, c: jnp.ndarray) -> jnp.ndarray:
        rho_offsets, n_rho = _rho_offsets(self.pipes)
        q_offsets, n_q = _q_offsets(self.pipes)
        c_rho = c[:n_rho]
        c_q   = c[n_rho:]

        R = jnp.zeros((n_q, n_q))

        for e, pipe in enumerate(self.pipes):
            h = pipe.element_length
            for k in range(pipe.n_elements):
                rho_ek = c_rho[rho_offsets[e] + k]
                q_in  = c_q[q_offsets[e] + k]
                q_out = c_q[q_offsets[e] + k + 1]
                base = q_offsets[e] + k

                for i in range(2):
                    for j in range(2):
                        val = (pipe.gamma / (rho_ek * rho_ek)) * self._integrate_abs_v_ri_rj(q_in, q_out, h, i, j)
                        # adjacent elements share the flux coefficient at their common
                        # interior node, so contributions must accumulate
                        R = R.at[base + i, base + j].add(val)

        return R

class R_pip:
    def __init__(self, nodes: Sequence[Node], pipes: Sequence[Pipe], basis_function: R_PiecewiseLinearContinuous):
        self.nodes = nodes
        self.pipes = pipes
        self.basis_function = basis_function
        # define static R_mo_q to later build the actual R_pip with c
        self._R_q_builder = R_mo_q(nodes, pipes, basis_function)

    def build(self, c: jnp.ndarray) -> jnp.ndarray:
        _, n_rho = _rho_offsets(self.pipes)
        _, n_q = _q_offsets(self.pipes)
        R_q = self._R_q_builder.build(c)

        R = jnp.zeros((n_rho + n_q, n_rho + n_q))
        R = R.at[n_rho:, n_rho:].set(R_q)
        return R