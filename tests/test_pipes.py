import pytest

from gasnetwork.topology import Node, Pipe


def test_pipe_default_flux_refinement():
    node1 = Node(id=1)
    node2 = Node(id=2)
    pipe = Pipe(id=1, innode=node1, outnode=node2, length=1000.0)

    assert pipe.flux_refinement == 2
    assert pipe.n_fluxes == 2
    assert pipe.n_elements == 1
    assert pipe.element_length == 1000.0


def test_pipe_flux_refinement_derived_properties():
    node1 = Node(id=1)
    node2 = Node(id=2)
    pipe = Pipe(id=1, innode=node1, outnode=node2, length=900.0, flux_refinement=4)

    assert pipe.n_fluxes == 4
    assert pipe.n_elements == 3
    assert pipe.element_length == 300.0


def test_pipe_flux_refinement_below_two_raises():
    node1 = Node(id=1)
    node2 = Node(id=2)
    with pytest.raises(ValueError):
        Pipe(id=1, innode=node1, outnode=node2, flux_refinement=1)
