"""Regression checks for bugs identified during the Pyright audit."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np

import particles
from particles import distributions as dists
from particles import hilbert, kalman, resampling as rs, rqmc, smc_samplers, smoothing
from particles.state_space_models import Bootstrap


class NumericalRegressions(unittest.TestCase):
    def test_ssp_singleton_rounding_correction(self):
        implementations = (rs.ssp, rs.ssp.__wrapped__.py_func)
        for implementation in implementations:
            for weight in (1.0, np.nextafter(1.0, 0.0)):
                for M in (0, 1, 5, 10):
                    with self.subTest(implementation=implementation, weight=weight, M=M):
                        result = implementation(np.array([weight]), M=M)
                        np.testing.assert_array_equal(result, np.zeros(M, dtype=int))
            # The correction does not normalize arbitrary invalid weights.
            with self.assertRaisesRegex(ValueError, "wrong size for output"):
                implementation(np.array([0.5]), M=1)

    def test_ssp_preserves_offspring_count_bounds(self):
        for W in (np.array([0.5, 0.5]), np.array([0.0, 1.0]),
                  np.array([0.25, 0.25, 0.5])):
            for M in (0, 1, 5, 17):
                with self.subTest(W=W, M=M):
                    result = rs.ssp(W, M=M)
                    self.assertEqual(result.size, M)
                    self.assertTrue(np.all((result >= 0) & (result < len(W))))
                    counts = np.bincount(result, minlength=len(W))
                    self.assertTrue(np.all(counts >= np.floor(M * W)))
                    self.assertTrue(np.all(counts <= np.ceil(M * W)))

    def test_tempering_bridge_warns_for_ignored_time_indices(self):
        class FullTarget(smc_samplers.TemperingBridge):
            def logtarget(self, theta):
                return -0.5 * (theta - 2.0) ** 2

        prior = dists.Normal()
        bridge = FullTarget(base_dist=prior)
        theta = np.array([-0.5, 0.0, 1.5])
        target = bridge.logtarget(theta)
        for method, expected in (
            (bridge.logpost, target),
            (bridge.loglik, target - prior.logpdf(theta)),
        ):
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                np.testing.assert_allclose(method(theta), expected)
                np.testing.assert_allclose(method(theta, t=None), expected)
            self.assertEqual(len(observed), 0)
            for t in (0, -1, 2):
                with self.subTest(method=method, t=t):
                    with self.assertWarnsRegex(RuntimeWarning, "t is ignored"):
                        actual = method(theta, t=t)
                    np.testing.assert_allclose(actual, expected)

    def test_likelihood_accumulates_with_and_without_resampling(self):
        class ConstantWeights(particles.FeynmanKac):
            def M0(self, N):
                return np.zeros(N)

            def M(self, t, xp):
                return xp

            def logG(self, t, xp, x):
                return np.full(len(x), np.log(t + 2.0))

            def time_to_resample(self, smc):
                return self.force_resampling

        for force_resampling in (False, True):
            with self.subTest(force_resampling=force_resampling):
                fk = ConstantWeights(T=3)
                fk.force_resampling = force_resampling
                pf = particles.SMC(fk=fk, N=4)
                pf.run()
                np.testing.assert_allclose(pf.summaries.logLts, np.log([2, 6, 24]))

    def test_trajectory_extraction_preserves_empty_and_short_histories(self):
        hist = smoothing.ParticleHistory(fk=None, qmc=False)
        self.assertEqual(hist.extract_one_trajectory(), [])
        hist.X = [np.array([9.0])]
        hist.wgts = [rs.Weights(lw=np.zeros(1))]
        hist.A = [None]
        self.assertEqual(hist.extract_one_trajectory(), [9.0])

        hist.X = [np.array([10, 20]), np.array([30, 40]), np.array([50, 60])]
        hist.wgts = [rs.Weights(lw=np.zeros(2)) for _ in range(3)]
        hist.A = [None, np.array([1, 0]), np.array([0, 1])]
        with patch.object(rs, "multinomial_once", return_value=1):
            self.assertEqual(hist.extract_one_trajectory(), [10, 40, 60])

    def test_unknown_ffbs_submethod_raises_a_clear_error(self):
        fk = Bootstrap(ssm=kalman.LinearGauss(), data=[0.0, 0.0])
        for method in ("FFBS_unknown", "FFBS"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "Unknown FFBS smoothing method"):
                    smoothing.smoothing_worker(method=method, fk=fk, N=4)

    def test_hilbert_round_trip(self):
        for dimension in (2, 3):
            for index in (0, 1, 3, 4, 15, 63):
                with self.subTest(dimension=dimension, index=index):
                    coordinates = hilbert.int_to_Hilbert(index, dimension)
                    self.assertEqual(hilbert.Hilbert_to_int(coordinates), index)

    def test_latin_hypercube_strata(self):
        n, dimension = 16, 3
        samples = rqmc.latin(n, dimension)
        self.assertEqual(samples.shape, (n, dimension))
        self.assertTrue(np.all((samples > 0.0) & (samples < 1.0)))
        for column in samples.T:
            np.testing.assert_array_equal(
                np.sort((n * column).astype(int)), np.arange(n)
            )

    def test_qmc_smoothing_worker(self):
        fk = Bootstrap(ssm=kalman.LinearGauss(), data=[0.0, 0.0, 0.0])
        result = smoothing.smoothing_worker(
            method="FFBS_QMC",
            N=16,
            fk=fk,
            add_func=lambda t, x, xf: x + xf,
        )
        self.assertEqual(result["est"].shape, (2,))
        self.assertTrue(np.all(np.isfinite(result["est"])))

    def test_multivariate_normal_dimensions(self):
        for loc in (0.0, np.array(0.0)):
            with self.subTest(loc=loc):
                with self.assertRaisesRegex(ValueError, "specify cov"):
                    dists.MvNormal(loc=loc)
        inferred = dists.MvNormal(loc=np.zeros(2))
        explicit = dists.MvNormal(cov=np.eye(2))
        np.testing.assert_array_equal(inferred.cov, explicit.cov)
        np.testing.assert_allclose(
            inferred.logpdf(np.ones(2)), explicit.logpdf(np.ones(2))
        )
        self.assertEqual(explicit.rvs(size=5).shape, (5, 2))


if __name__ == "__main__":
    unittest.main()
