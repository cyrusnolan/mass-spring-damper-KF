import numpy as np
import unittest
import kf as python_kf
from parameterized import parameterized_class

@parameterized_class([
    { "KF": python_kf.KalmanFilter }
])
class TestKalmanFilter(unittest.TestCase):
    # def setUp(self):
    #     pass

    def test_shape(self):
        nx = 6 # num states
        nu = 1 # num inputs
        nz = 2 # num sensors
        F = np.ones((nx, nx))
        G = np.ones((nx, nu))
        H = np.ones((nz, nx))
        Q = np.ones((nu, nu))
        R = np.ones((nz, nz))
        X0 = np.ones((nx, 1))
        P0 = np.ones((nx, nx))
        self.kf = self.KF(X0, P0, F, G, H, Q, R)
        kf = self.kf
        kf.predict(u=0)
        self.assertAlmostEqual(kf.F.shape, (nx, nx))
        self.assertAlmostEqual(kf.G.shape, (nx, nu))
        self.assertAlmostEqual(kf.H.shape, (nz, nx))
        self.assertAlmostEqual(kf.Q.shape, (nu, nu))
        self.assertAlmostEqual(kf.R.shape, (nz, nz))
        self.assertEqual(kf.Pu.shape, (nx, nx))
        self.assertEqual(kf.xu.shape, (nx, 1))
        self.assertEqual(kf.Pp.shape, (nx, nx))
        self.assertEqual(kf.xp.shape, (nx, 1))
