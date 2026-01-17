"""
Tests for functionality checks in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import numpy as np


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()

    w = 12.
    h = 8.
    dx = 0.5
    dy = 0.5
    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)

    d = 5.
    T_cold = 280.
    T_hot = 600.
    solver.initialize_physical_parameters(d=d, T_cold=T_cold, T_hot=T_hot)

    # Manual calculation of expected dt from the formula used in diffusion2d.py
    # dt = dx^2 * dy^2 / (2 * D * (dx^2 + dy^2))
    dx2 = dx * dx
    dy2 = dy * dy
    expected_dt = dx2 * dy2 / (2 * d * (dx2 + dy2))

    assert np.isclose(solver.dt, expected_dt), 'dt not computed correctly'


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()

    w = 10.
    h = 9.
    dx = 0.2
    dy = 0.2
    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    
    d = 4.
    T_cold = 150.
    T_hot = 500.
    solver.initialize_physical_parameters(d=d, T_cold=T_cold, T_hot=T_hot)

    u = solver.set_initial_condition()

    # Manual calculation of expected initial condition u
    nx = int (w / dx)
    ny = int (h / dy)
    expected_u = T_cold * np.ones((nx, ny))

    for i in range(nx):
        for j in range(ny):
            r = 2
            cx = 5
            cy = 5
            r2 = r ** 2
            p2 = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
            if p2 < r2:
                expected_u[i, j] = T_hot

    np.testing.assert_array_equal(u, expected_u, 'Initial condition u not set correctly')