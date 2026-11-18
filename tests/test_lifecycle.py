"""Compatibility checks for optional dependencies and history selection."""

import unittest

import numpy as np

import particles
from particles import collectors, distributions as dists, hmm, kalman, mcmc
from particles import nested, smc_samplers as ssps, smoothing
from particles.state_space_models import AuxiliaryPF, Bootstrap, StochVol


class DeterministicFK(particles.FeynmanKac):
    def __init__(self):
        super().__init__(T=2)
        self.parents = []

    def M0(self, N):
        return np.arange(N, dtype=float)

    def M(self, t, xp):
        return xp + 1.0

    def logG(self, t, xp, x):
        self.parents.append(xp)
        return np.zeros(len(x))


class GaussianLocation(ssps.StaticModel):
    def logpyt(self, theta, t):
        return dists.Normal(loc=theta["mu"]).logpdf(self.data[t])


class BoundedLinearGauss(kalman.LinearGauss):
    def upper_bound_log_pt(self, t):
        return -0.5 * np.log(2.0 * np.pi) - np.log(self.sigmaX)


class FixedParameterGibbs(mcmc.ParticleGibbs):
    def update_theta(self, theta, x):
        return theta


def location_model():
    return GaussianLocation(
        data=[0.0, 0.1, -0.1], prior=dists.StructDist({"mu": dists.Normal()})
    )


class SeededTestCase(unittest.TestCase):
    def setUp(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        np.random.seed(1234)


class DependencyLifecycle(SeededTestCase):
    def test_deferred_smc_and_time_zero_state(self):
        pf = particles.SMC(N=4)
        self.assertIsNone(pf.X)
        self.assertIsNone(pf.Xp)
        self.assertIsNone(pf.A)
        self.assertIsNone(pf.aux)
        with self.assertRaisesRegex(ValueError, "fk must be set"):
            next(pf)
        self.assertEqual(pf.t, 0)
        fk = DeterministicFK()
        self.assertFalse(fk.isAPF)
        pf.fk = fk
        next(pf)
        self.assertIsNone(fk.parents[0])
        self.assertIsNone(pf.Xp)
        self.assertIsNone(pf.A)
        next(pf)
        np.testing.assert_array_equal(pf.X, np.arange(4) + 1.0)
        np.testing.assert_array_equal(fk.parents[1], np.arange(4))
        self.assertIsNotNone(pf.aux)
        with self.assertRaises(StopIteration):
            next(pf)

    def test_stateless_particles_remain_supported_without_resampling(self):
        class StatelessFK(particles.FeynmanKac):
            def M0(self, N):
                self.N = N
                return None

            def M(self, t, xp):
                return None

            def logG(self, t, xp, x):
                return np.zeros(self.N)

        pf = particles.SMC(fk=StatelessFK(T=2), N=4)
        pf.run()
        self.assertEqual(pf.t, 2)
        self.assertIsNone(pf.X)
        self.assertIsNone(pf.Xp)
        self.assertEqual(pf.logLt, 0.0)

    def test_deferred_exact_filters(self):
        kf = kalman.Kalman()
        with self.assertRaisesRegex(ValueError, "data must be set"):
            next(kf)
        kf.data = [0.0, 0.1]
        with self.assertRaisesRegex(ValueError, "ssm must be set"):
            next(kf)
        kf.ssm = kalman.LinearGauss()
        kf.filter()
        kf.smoother()
        self.assertEqual(len(kf.smth), 2)

        bw = hmm.BaumWelch()
        bw.data = [0.0, 0.1]
        with self.assertRaisesRegex(ValueError, "hmm must be set"):
            next(bw)
        bw.hmm = hmm.GaussianHMM(
            trans_mat=np.array([[0.9, 0.1], [0.2, 0.8]]),
            mus=np.array([0.0, 1.0]), sigmas=np.ones(2),
        )
        bw.run()
        self.assertEqual(bw.sample(N=3).shape, (2, 3))

    def test_empty_exact_filters_do_not_require_a_model(self):
        kalman.Kalman(data=[]).filter()
        hmm.BaumWelch(data=[]).forward()

    def test_deferred_model_prior_and_missing_distribution(self):
        model = GaussianLocation(data=[0.0])
        theta = np.zeros(2, dtype=[("mu", float)])
        with self.assertRaisesRegex(ValueError, "prior must be set"):
            model.logpost(theta)
        model.prior = dists.StructDist({"mu": dists.Normal()})
        self.assertTrue(np.all(np.isfinite(model.logpost(theta))))

        mix = dists.MixMissing(pmiss=0.2)
        with self.assertRaisesRegex(ValueError, "base_dist must be set"):
            mix.logpdf(np.array([0.0]))
        mix.base_dist = dists.Normal()
        np.testing.assert_allclose(mix.logpdf(np.array([np.nan])), np.log(0.2))

    def test_deferred_nested_model(self):
        sampler = nested.NestedSampling(N=4)
        with self.assertRaisesRegex(ValueError, "model must be set"):
            sampler.setup()
        sampler.model = location_model()
        sampler.setup()
        self.assertEqual(sampler.x.N, 4)
        self.assertTrue(np.all(np.isfinite(sampler.x.llik)))

    def test_deferred_importance_model_with_explicit_proposal(self):
        proposal = dists.StructDist({"mu": dists.Normal()})
        sampler = ssps.ImportanceSampler(proposal=proposal)
        sampler.model = location_model()
        sampler.run(N=8)
        self.assertTrue(np.isfinite(sampler.log_norm_cst))

    def test_structured_particles_and_deferred_ibis_model(self):
        fk = ssps.IBIS(len_chain=3)
        fk.model = location_model()
        pf = particles.SMC(fk=fk, N=8, collect=[collectors.Moments()])
        pf.run()
        self.assertIsInstance(pf.X, ssps.ThetaParticles)
        self.assertEqual(len(pf.summaries.moments), 3)
        self.assertTrue(np.isfinite(pf.logLt))

    def test_copy_masks_and_empty_collections(self):
        ssps.FancyList([]).copyto(ssps.FancyList([]))
        ssps.ThetaParticles().copyto(ssps.ThetaParticles())
        dst = ssps.ThetaParticles(theta=np.zeros((3, 2)))
        src = ssps.ThetaParticles(theta=np.ones((3, 2)))
        dst.copyto(src, where=np.array([True, False, True]))
        np.testing.assert_array_equal(dst.theta, [[1, 1], [0, 0], [1, 1]])


class HistoryCompatibility(SeededTestCase):
    def test_history_modes_remain_distinct(self):
        cases = [
            (False, type(None), None),
            (True, smoothing.ParticleHistory, 2),
            (0, smoothing.RollingParticleHistory, 0),
            (1, smoothing.RollingParticleHistory, 1),
            (lambda t: t == 0, smoothing.PartialParticleHistory, 1),
        ]
        for option, expected_type, length in cases:
            with self.subTest(option=option):
                pf = particles.SMC(fk=DeterministicFK(), N=4, store_history=option)
                self.assertIs(type(pf.hist), expected_type)
                pf.run()
                if pf.hist is not None:
                    self.assertEqual(len(pf.hist.X), length)

    def test_missing_information_history_is_rejected(self):
        fk = Bootstrap(ssm=kalman.LinearGauss(), data=[0.0, 0.1, -0.1])
        pf = particles.SMC(fk=fk, N=8, store_history=True)
        pf.run()
        info = particles.SMC(fk=fk, N=8, store_history=False)
        info.run()
        with self.assertRaisesRegex(ValueError, "information-filter history"):
            pf.hist.two_filter_smoothing(
                0, info, lambda x, xf: x + xf, lambda x: np.zeros_like(x)
            )

    def test_partial_information_history_retains_supported_uses(self):
        # Two-filter estimates need stored states and weights, not backward methods.
        fk = Bootstrap(ssm=BoundedLinearGauss(), data=[0.0, 0.1, -0.1])
        pf = particles.SMC(fk=fk, N=8, store_history=True)
        info = particles.SMC(fk=fk, N=8, store_history=lambda t: True)
        pf.run()
        info.run()
        result = pf.hist.two_filter_smoothing(
            0, info, lambda x, xf: x + xf, lambda x: np.zeros_like(x)
        )
        self.assertTrue(np.isfinite(result))

    def test_complete_history_in_two_filter_workers(self):
        fk = Bootstrap(ssm=BoundedLinearGauss(), data=[0.0, 0.1, -0.1])
        for method in ("two-filter_ON2", "two-filter_ON", "two-filter_ON_prop"):
            with self.subTest(method=method):
                result = smoothing.smoothing_worker(
                    method=method, fk=fk, N=16,
                    add_func=lambda t, x, xf: x + xf,
                    log_gamma=lambda x: np.zeros_like(x),
                )
                self.assertEqual(result["est"].shape, (2,))
                self.assertTrue(np.all(np.isfinite(result["est"])))

    def test_unused_additive_function_remains_optional(self):
        fk = Bootstrap(ssm=kalman.LinearGauss(), data=[0.0])
        result = smoothing.smoothing_worker(method="FFBS_ON2", fk=fk, N=4)
        self.assertEqual(result["est"].shape, (0,))

    def test_conditional_filter_retains_the_reference_path(self):
        fk = Bootstrap(ssm=kalman.LinearGauss(), data=[0.0, 0.1])
        pf = mcmc.CSMC(fk=fk, N=8)
        pf.xstar = [np.array(0.25), np.array(-0.5)]
        pf.run()
        self.assertIsInstance(pf.hist, smoothing.ParticleHistory)
        self.assertEqual(pf.hist.X[0][0], 0.25)
        self.assertEqual(pf.hist.X[1][0], -0.5)
        self.assertEqual(pf.A[0], 0)


class SamplerCompatibility(SeededTestCase):
    def prior(self):
        return dists.StructDist({"mu": dists.Normal(loc=-1.0, scale=0.2)})

    def test_auxiliary_filters_and_quasi_monte_carlo(self):
        for qmc in (False, True):
            with self.subTest(qmc=qmc):
                fk = AuxiliaryPF(ssm=StochVol(), data=[0.1, 0.2, -0.1])
                self.assertTrue(fk.isAPF)
                pf = particles.SMC(fk=fk, N=16, qmc=qmc, ESSrmin=1.0)
                pf.run()
                self.assertTrue(np.isfinite(pf.logLt))
                self.assertIsNotNone(pf.aux)

    def test_particle_marginal_metropolis(self):
        sampler = mcmc.PMMH(
            ssm_cls=StochVol, prior=self.prior(), data=[0.1, 0.2, -0.1],
            niter=4, Nx=8,
        )
        sampler.run()
        self.assertEqual(sampler.chain.theta.shape, (4,))
        self.assertTrue(np.all(np.isfinite(sampler.chain.lpost)))

    def test_particle_gibbs_history_consumers(self):
        for backward_step in (False, True):
            with self.subTest(backward_step=backward_step):
                sampler = FixedParameterGibbs(
                    ssm_cls=StochVol, prior=self.prior(),
                    data=[0.1, 0.2, -0.1], niter=3, Nx=8, verbose=0,
                    store_x=True, backward_step=backward_step,
                    regenerate_data=backward_step,
                )
                sampler.run()
                self.assertEqual(sampler.chain.x.shape, (3, 3))
                self.assertTrue(np.all(np.isfinite(sampler.chain.x)))

    def test_smc_squared(self):
        fk = ssps.SMC2(
            ssm_cls=StochVol, prior=self.prior(), data=[0.1, 0.2, -0.1],
            init_Nx=8, len_chain=2, wastefree=False,
        )
        pf = particles.SMC(fk=fk, N=16)
        pf.run()
        self.assertTrue(np.isfinite(pf.logLt))
        self.assertEqual(len(pf.X.pfs), 16)


if __name__ == "__main__":
    unittest.main()
