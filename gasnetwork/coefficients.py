import jax.numpy as jnp
from dataclasses import dataclass

"""
This file contains an immutable dataclass containing all the Runge-Kutta coefficients of the methods
in use, which are:
    - Radau IIA         (for the monolithic implementation)
    - Lobatto IIIA      (for the partitioned implementation)
    - Lobatto IIIB      (for the partitioned implementation)
"""

@dataclass(frozen=True)
class ButcherTableau:
    A: jnp.ndarray          # shape (s, s)
    b: jnp.ndarray          # shape(s, )
    c: jnp.ndarray          # shape (s, )
    s: int                  # number of stages

sqrt6 = jnp.sqrt(6)
sqrt5 = jnp.sqrt(5)

RADAU_IIA = {
    1: ButcherTableau(
        A=jnp.array([[1.0]]),
        b=jnp.array([1.0]),
        c=jnp.array([1.0]),
        s=1,
    ),
    2: ButcherTableau(
        A=jnp.array([
            [5.0/12.0, -1.0/12.0],
            [3.0/4.0,   1.0/4.0],
        ]),
        b=jnp.array([3.0/4.0, 1.0/4.0]),
        c=jnp.array([1.0/3.0, 1.0]),
        s=2,
    ),
    3: ButcherTableau(
        A=jnp.array([
            [(88.0 - 7.0*sqrt6)/360.0,   (296.0 - 169.0*sqrt6)/1800.0, (-2.0 + 3.0*sqrt6)/225.0],
            [(296.0 + 169.0*sqrt6)/1800.0, (88.0 + 7.0*sqrt6)/360.0,   (-2.0 - 3.0*sqrt6)/225.0],
            [(16.0 - sqrt6)/36.0,          (16.0 + sqrt6)/36.0,        1.0/9.0],
        ]),
        b=jnp.array([(16.0 - sqrt6)/36.0, (16.0 + sqrt6)/36.0, 1.0/9.0]),
        c=jnp.array([(4.0 - sqrt6)/10.0, (4.0 + sqrt6)/10.0, 1.0]),
        s=3,
    ),
}

LOBATTO_IIIA = {
    2: ButcherTableau(
        A=jnp.array([
            [0.0, 0.0],
            [0.5, 0.5],
        ]),
        b=jnp.array([0.5, 0.5]),
        c=jnp.array([0.0, 1.0]),
        s=2,
    ),
    3: ButcherTableau(
        A=jnp.array([
            [0.0,        0.0,       0.0],
            [5.0/24.0,   1.0/3.0,  -1.0/24.0],
            [1.0/6.0,    2.0/3.0,   1.0/6.0],
        ]),
        b=jnp.array([1.0/6.0, 2.0/3.0, 1.0/6.0]),
        c=jnp.array([0.0, 0.5, 1.0]),
        s=3,
    ),
    4: ButcherTableau(
        A=jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [(11.0 + sqrt5) / 120.0, (25.0 - sqrt5) / 120.0, (25.0 - 13.0*sqrt5) / 120.0, (-1.0 + sqrt5) / 120.0],
            [(11.0 - sqrt5) / 120.0, (25.0 + 13.0*sqrt5) / 120.0, (25.0 + sqrt5) / 120.0, (-1.0 - sqrt5) / 120.0],
            [1.0/12.0, 5.0/12.0, 5.0/12.0, 1.0/12.0],
        ]),
        b=jnp.array([1.0/12.0, 5.0/12.0, 5.0/12.0, 1.0/12.0]),
        c=jnp.array([0.0, (5.0 - sqrt5) / 10.0, (5.0 + sqrt5) / 10.0, 1.0]),
        s=4,
    ),
}

LOBATTO_IIIB = {
    2: ButcherTableau(
        A=jnp.array([
            [0.5, 0.0],
            [0.5, 0.0],
        ]),
        b=jnp.array([0.5, 0.5]),
        c=jnp.array([0.0, 1.0]),
        s=2,
    ),
    3: ButcherTableau(
        A=jnp.array([
            [1.0/6.0, -1.0/6.0, 0.0],
            [1.0/6.0,  1.0/3.0, 0.0],
            [1.0/6.0,  5.0/6.0, 0.0],
        ]),
        b=jnp.array([1.0/6.0, 2.0/3.0, 1.0/6.0]),
        c=jnp.array([0.0, 0.5, 1.0]),
        s=3,
    ),
    4: ButcherTableau(
        A=jnp.array([
            [1.0/12.0, (-1.0 - sqrt5) / 24.0, (-1.0 + sqrt5) / 24.0, 0.0],
            [1.0/12.0, (25.0 + sqrt5) / 120.0, (25.0 - 13.0*sqrt5) / 120.0, 0.0],
            [1.0/12.0, (25.0 + 13.0*sqrt5) / 120.0, (25.0 - sqrt5) / 120.0, 0.0],
            [1.0/12.0, (11.0 - sqrt5) / 24.0, (11.0 + sqrt5) / 24.0, 0.0],
        ]),
        b=jnp.array([1.0/12.0, 5.0/12.0, 5.0/12.0, 1.0/12.0]),
        c=jnp.array([0.0, (5.0 - sqrt5) / 10.0, (5.0 + sqrt5) / 10.0, 1.0]),
        s=4,
    ),
}

LOBATTO_IIIC = {
    2: ButcherTableau(
        A=jnp.array([
            [ 1.0/2.0, -1.0/2.0],
            [ 1.0/2.0,  1.0/2.0],
        ]),
        b=jnp.array([1.0/2.0, 1.0/2.0]),
        c=jnp.array([0.0, 1.0]),
        s=2,
    ),
    3: ButcherTableau(
        A=jnp.array([
            [ 1.0/6.0, -1.0/3.0,  1.0/6.0],
            [ 1.0/6.0,  5.0/12.0, -1.0/12.0],
            [ 1.0/6.0,  2.0/3.0,  1.0/6.0],
        ]),
        b=jnp.array([1.0/6.0, 2.0/3.0, 1.0/6.0]),
        c=jnp.array([0.0, 1.0/2.0, 1.0]),
        s=3,
    ),
    4: ButcherTableau(
        A=jnp.array([
            [ 1.0/12.0, -sqrt5/12.0,        sqrt5/12.0,       -1.0/12.0],
            [ 1.0/12.0,  1.0/4.0,           (10.0 - 7.0*sqrt5)/60.0, sqrt5/60.0],
            [ 1.0/12.0,  (10.0 + 7.0*sqrt5)/60.0, 1.0/4.0,     -sqrt5/60.0],
            [ 1.0/12.0,  5.0/12.0,          5.0/12.0,          1.0/12.0],
        ]),
        b=jnp.array([1.0/12.0, 5.0/12.0, 5.0/12.0, 1.0/12.0]),
        c=jnp.array([0.0, (5.0 - sqrt5)/10.0, (5.0 + sqrt5)/10.0, 1.0]),
        s=4,
    ),
}

def get_tableau(family: str, s: int) -> ButcherTableau:
    """
    Simple function fort retreiving butcher tableaus from only specifying method-name and 
    number of stages. 
    Added a fallback in case method-name is not in all lower-case.
    """
    dict = {"radau_iia": RADAU_IIA, "lobatto_iiia": LOBATTO_IIIA, "lobatto_iiib": LOBATTO_IIIB, "lobatto_iiic": LOBATTO_IIIC}
    key = family.strip().lower()
    if key not in dict.keys():
        raise ValueError(f"Unknown family {key}, choose from {sorted(dict)}")
    if s not in dict[key].keys():
        raise ValueError(f"No {key} tableau for s = {s}. Available s: {sorted(dict[key])}")
    return dict[key][s]