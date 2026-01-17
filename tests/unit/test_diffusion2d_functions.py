"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import numpy as np


def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()

    w = 12.
    h = 8.
    dx = 0.5
    dy = 0.4

    expected_nx = int(w / dx)
    expected_ny = int(h / dy)

    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)

    assert solver.nx == expected_nx, 'nx not correctly initialized, wrong value'
    assert solver.ny == expected_ny, 'ny not correctly initialized, wrong value'


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()

    solver.dx = 0.14
    solver.dy = 0.16

    d = 6.
    T_cold = 290.
    T_hot = 450.

    dx2 = solver.dx * solver.dx
    dy2 = solver.dy * solver.dy
    expected_dt = dx2 * dy2 / (2 * d * (dx2 + dy2))

    solver.initialize_physical_parameters(d=d, T_cold=T_cold, T_hot=T_hot)

    assert solver.D == d, 'D not correctly initialized, wrong value'
    assert solver.T_cold == T_cold, 'T_cold not correctly initialized, wrong value'
    assert solver.T_hot == T_hot, 'T_hot not correctly initialized, wrong value'
    assert solver.dt == expected_dt, 'dt not correctly initialized, wrong value'


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()

    solver.w = 10.
    solver.h = 10.
    solver.dx = 0.5
    solver.dy = 0.5
    solver.D = 7.
    solver.T_cold = 100.
    solver.T_hot = 500.

    solver.nx = int(solver.w / solver.dx)
    solver.ny = int(solver.h / solver.dy)

    u = solver.set_initial_condition()

    # Check shape
    assert u.shape == (solver.nx, solver.ny), 'Initial condition u has wrong shape'
    
    # Check whether a point far from the center is cold
    assert u[0,0] == solver.T_cold, 'Initial condition u far from center not cold'

    # Check whether the center point is hot
    assert u[10,10] == solver.T_hot, 'Initial condition u at center not hot'

    # Check whether both values appear
    assert np.any(u == solver.T_cold), 'Initial condition u does not contain T_cold'
    assert np.any(u == solver.T_hot), 'Initial condition u does not contain T_hot'