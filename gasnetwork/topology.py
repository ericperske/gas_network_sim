import numpy as np
from typing import Sequence

from gasnetwork.errors import NetworkTopologyError
    
class Node:
    """A node in the gas network."""

    def __init__(self,
            id: str | int, 
            h: float | None = None,     # optional enthalpy value for the node
            flux: float | None = None,  # optional mass flux value for the node
        ) -> None:
        if h is not None and flux is not None:
            raise ValueError(f"Node {id} cannot have both enthalpy and flux values defined. Please specify only one of them.")

        self.id = id
        self.enthalpy = h
        self.flux = flux

    @property
    def is_enthalpy_node(self) -> bool:
        return self.enthalpy is not None

    @property
    def is_flux_node(self) -> bool:
        return self.flux is not None

    @property
    def is_interior(self) -> bool:
        return self.enthalpy is None and self.flux is None

class Pipe:
    """A pipe in the gas network."""

    def __init__(self,
            id: str | int,
            innode: Node,
            outnode: Node,
            length: float=1000.0,
            diameter: float=1.0,
            angle: float=0.0,
            roughness: float=1e-5,
            flux_refinement: int=2,
        ) -> None:
        if flux_refinement < 2:
            raise ValueError(f"Pipe {id}: flux_refinement must be >= 2 (got {flux_refinement}).")

        self.id = id
        self.innode = innode
        self.outnode = outnode
        self.length = length
        self.diameter = diameter
        self.angle = angle
        self.roughness = roughness
        self.flux_refinement = flux_refinement

        self.cross_sectional_area = np.pi * (diameter ** 2) / 4

        nikuradse_friction = ((-2) * np.log10(roughness / (3.71 * diameter))) ** (-2)

        self.gamma = nikuradse_friction / (2 * diameter * self.cross_sectional_area)

    @property
    def n_fluxes(self) -> int:
        """Number of local flux (hat function) coefficients on this pipe."""
        return self.flux_refinement

    @property
    def n_elements(self) -> int:
        """Number of local density (piecewise-constant) coefficients / sub-elements on this pipe."""
        return self.flux_refinement - 1

    @property
    def element_length(self) -> float:
        """Length of each uniform sub-element."""
        return self.length / self.n_elements

class IncidenceMatrix:
    def __init__(self, 
            nodes: Sequence[Node],
            pipes: Sequence[Pipe],
        ):
        self.nodes = nodes
        self.pipes = pipes
    
    def build(self) -> np.ndarray:
        node_ids = [node.id for node in self.nodes]
        pipe_ids = [pipe.id for pipe in self.pipes]
        if len(node_ids) != len(set(node_ids)):
            raise NetworkTopologyError("Duplicate node IDs found. Node IDs must be unique.")
        if len(pipe_ids) != len(set(pipe_ids)):
            raise NetworkTopologyError("Duplicate pipe IDs found. Pipe IDs must be unique.")

        node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        src_idx = np.empty(len(self.pipes), dtype=int)
        dst_idx = np.empty(len(self.pipes), dtype=int)
        degree = np.zeros(len(self.nodes), dtype=int)
        adj = [[] for _ in self.nodes]
        seen_edges: set[tuple[int, int]] = set()

        for col, pipe in enumerate(self.pipes):
            in_id, out_id = pipe.innode.id, pipe.outnode.id
            if in_id not in node_id_to_index or out_id not in node_id_to_index:
                raise NetworkTopologyError(f"Pipe '{pipe.id}' references unknown node(s): {in_id!r} -> {out_id!r}.")

            u, v = node_id_to_index[in_id], node_id_to_index[out_id]
            if u == v:
                raise NetworkTopologyError(f"Pipe '{pipe.id}' is a self-loop at node '{in_id}'.")

            e = (u, v) if u < v else (v, u)
            if e in seen_edges:
                raise NetworkTopologyError("Found parallel edges in the network. Please adjust network topology to ensure no parallel edges.")
            seen_edges.add(e)

            src_idx[col], dst_idx[col] = u, v
            degree[u] += 1
            degree[v] += 1
            adj[u].append(v)
            adj[v].append(u)

        hanging_nodes = int(np.count_nonzero(degree == 0)) if len(self.nodes) else 0
        if hanging_nodes:
            raise NetworkTopologyError(f"Found {hanging_nodes} hanging nodes. Please adjust network topology to ensure connectivity.")

        if self.nodes:
            visited = set()
            stack = [0]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                stack.extend(v for v in adj[u] if v not in visited)
            if len(visited) != len(self.nodes):
                raise NetworkTopologyError("Network is disconnected. Please adjust topology to ensure one connected component.")

        matrix = np.zeros((len(self.nodes), len(self.pipes)), dtype=float)
        cols = np.arange(len(self.pipes), dtype=int)
        matrix[src_idx, cols] = -1.0
        matrix[dst_idx, cols] = 1.0

        return matrix