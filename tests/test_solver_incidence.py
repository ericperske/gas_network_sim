import pytest
import numpy as np

from gasnetwork.topology import Node, Pipe, IncidenceMatrix
from gasnetwork.errors import NetworkTopologyError


def test_build_incidence_matrix():
    nodes = [
        Node(id="n1"),
        Node(id="n2"),
        Node(id="n3")
    ]
    pipes = [
        Pipe(id="p1", innode=nodes[0], outnode=nodes[1]),
        Pipe(id="p2", innode=nodes[1], outnode=nodes[2]),
    ]

    incidence_matrix = IncidenceMatrix(nodes, pipes)
    result = incidence_matrix.build()

    expected = np.array(
        [
            [-1.0, 0.0],
            [1.0, -1.0],
            [0.0, 1.0],
        ]
    )

    np.testing.assert_array_equal(result, expected)

def test_build_incidence_matrix_with_unconnected_node():
    nodes = [
        Node(id="n1"),
        Node(id="n2"),
        Node(id="n3")
    ]
    pipes = [
        Pipe(id="p1", innode=nodes[0], outnode=nodes[1]),
    ]

    incidence_matrix = IncidenceMatrix(nodes, pipes)
    with pytest.raises(NetworkTopologyError):
        incidence_matrix.build()

def test_build_incidence_matrix_with_parallel_edges():
    nodes = [
        Node(id="n1"),
        Node(id="n2"),
    ]
    pipes = [
        Pipe(id="p1", innode=nodes[0], outnode=nodes[1]),
        Pipe(id="p2", innode=nodes[0], outnode=nodes[1]),
    ]

    incidence_matrix = IncidenceMatrix(nodes, pipes)
    with pytest.raises(NetworkTopologyError):
        incidence_matrix.build()

def test_inverted_parallel_edges():
    nodes = [
        Node(id="n1"),
        Node(id="n2"),
    ]

    pipes = [
        Pipe(id="p1", innode=nodes[0], outnode=nodes[1]),
        Pipe(id="p2", innode=nodes[1], outnode=nodes[0]),
    ]

    incidence_matrix = IncidenceMatrix(nodes, pipes)
    with pytest.raises(NetworkTopologyError):
        incidence_matrix.build()

def test_duplicate_node_ids():
    nodes = [
        Node(id="n1"),
        Node(id="n1"),  # Duplicate ID
    ]
    pipes = []

    incidence_matrix = IncidenceMatrix(nodes, pipes)
    with pytest.raises(NetworkTopologyError):
        incidence_matrix.build()

def test_duplicate_pipe_ids():
    nodes = [
        Node(id="n1"),
        Node(id="n2"),
        Node(id="n3")
    ]
    pipes = [
        Pipe(id="p1", innode=nodes[0], outnode=nodes[1]),
        Pipe(id="p1", innode=nodes[1], outnode=nodes[2]),
    ]

    incidence_matrix = IncidenceMatrix(nodes, pipes)
    with pytest.raises(NetworkTopologyError):
        incidence_matrix.build()

def test_pipe_self_loop():
    nodes = [
        Node(id="n1"),
    ]
    pipes = [
        Pipe(id="p1", innode=nodes[0], outnode=nodes[0]),
    ]

    incidence_matrix = IncidenceMatrix(nodes, pipes)
    with pytest.raises(NetworkTopologyError):
        incidence_matrix.build()



if __name__ == "__main__":
    test_build_incidence_matrix()
    test_build_incidence_matrix_with_unconnected_node()
    test_build_incidence_matrix_with_parallel_edges()
    test_inverted_parallel_edges()
    test_duplicate_node_ids()
    test_duplicate_pipe_ids()
    test_pipe_self_loop()

    print("All tests passed!")