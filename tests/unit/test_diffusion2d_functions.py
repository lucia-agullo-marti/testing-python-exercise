"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import numpy as np
import unittest

class TestSolveDiffusion2D(unittest.TestCase):
    """
    Test class for SolveDiffusion2D
    """
    def setUp(self):
        """
        Setup function
        """
        self.solver = SolveDiffusion2D()

    def test_initialize_domain(self):
        """
        Check function SolveDiffusion2D.initialize_domain
        """

        w = 12.
        h = 8.
        dx = 0.5
        dy = 0.4

        expected_nx = int(w / dx)
        expected_ny = int(h / dy)

        self.solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)

        self.assertEqual(self.solver.nx, expected_nx, 'nx not correctly initialized, wrong value')
        self.assertEqual(self.solver.ny, expected_ny, 'ny not correctly initialized, wrong value')

    def test_initialize_physical_parameters(self):
        """
        Checks function SolveDiffusion2D.initialize_domain
        """
        self.solver.dx = 0.14
        self.solver.dy = 0.16

        d = 6.
        T_cold = 290.
        T_hot = 450.

        dx2 = self.solver.dx * self.solver.dx
        dy2 = self.solver.dy * self.solver.dy
        expected_dt = dx2 * dy2 / (2 * d * (dx2 + dy2))

        self.solver.initialize_physical_parameters(d=d, T_cold=T_cold, T_hot=T_hot)

        self.assertEqual(self.solver.D, d, 'D not correctly initialized, wrong value')
        self.assertEqual(self.solver.T_cold, T_cold, 'T_cold not correctly initialized, wrong value')
        self.assertEqual(self.solver.T_hot, T_hot, 'T_hot not correctly initialized, wrong value')
        self.assertEqual(self.solver.dt, expected_dt, 'dt not correctly initialized, wrong value')

    def test_set_initial_condition(self):
        """
        Checks function SolveDiffusion2D.get_initial_function
        """
        self.solver.w = 10.
        self.solver.h = 10.
        self.solver.dx = 0.5
        self.solver.dy = 0.5
        self.solver.D = 7.
        self.solver.T_cold = 100.
        self.solver.T_hot = 500.

        self.solver.nx = int(self.solver.w / self.solver.dx)
        self.solver.ny = int(self.solver.h / self.solver.dy)

        u = self.solver.set_initial_condition()

        # Check shape
        self.assertEqual(u.shape, (self.solver.nx, self.solver.ny), 'Initial condition u has wrong shape')
        
        # Check whether a point far from the center is cold
        self.assertEqual(u[0,0], self.solver.T_cold, 'Initial condition u far from center not cold')

        # Check whether the center point is hot
        self.assertEqual(u[10,10], self.solver.T_hot, 'Initial condition u at center not hot')

        # Check whether both values appear
        self.assertTrue(np.any(u == self.solver.T_cold), 'Initial condition u does not contain T_cold')
        self.assertTrue(np.any(u == self.solver.T_hot), 'Initial condition u does not contain T_hot')