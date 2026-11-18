"""Public argument compatibility for variable-selection likelihoods."""

import unittest

import numpy as np

from particles import binary_smc as binary
from particles import distributions as dists


class VariableSelectionArguments(unittest.TestCase):
    def setUp(self):
        predictors = np.column_stack(
            (np.ones(6), np.array([-1.0, 0.0, 1.0, 2.0, -2.0, 3.0]))
        )
        response = np.array([-0.2, 0.3, 1.1, 1.5, -0.7, 2.1])
        data = predictors, response
        prior = dists.IID(binary.Bernoulli(0.5), 2)
        self.models = (
            binary.BIC(data=data),
            binary.BayesianVS(data=data, prior=prior),
            binary.BayesianVS_gprior(data=data, prior=prior),
        )
        self.models[0].prior = prior
        self.gamma = binary.all_binary_words(2)

    def test_positional_and_both_keyword_forms_agree(self):
        original = self.gamma.copy()
        for model in self.models:
            expected = model.loglik(self.gamma)
            for t in (None, 0, 1):
                with self.subTest(model=type(model).__name__, t=t):
                    np.testing.assert_array_equal(model.loglik(self.gamma, t), expected)
                    np.testing.assert_array_equal(
                        model.loglik(theta=self.gamma, t=t), expected
                    )
                    np.testing.assert_array_equal(
                        model.loglik(gamma=self.gamma, t=t), expected
                    )
        np.testing.assert_array_equal(self.gamma, original)

    def test_supplying_both_names_is_rejected(self):
        for model in self.models:
            for theta, gamma in (
                (self.gamma, self.gamma),
                (None, self.gamma),
                (self.gamma, None),
                (None, None),
            ):
                with self.subTest(model=type(model).__name__, theta=theta, gamma=gamma):
                    with self.assertRaisesRegex(TypeError, "not both"):
                        model.loglik(theta=theta, gamma=gamma)
            with self.assertRaisesRegex(TypeError, "not both"):
                model.loglik(self.gamma, gamma=self.gamma)

    def test_omitting_both_names_is_rejected(self):
        for model in self.models:
            with self.subTest(model=type(model).__name__):
                with self.assertRaisesRegex(TypeError, "requires 'theta'"):
                    model.loglik()
                with self.assertRaisesRegex(TypeError, "requires 'theta'"):
                    model.loglik(t=0)

    def test_generic_logposterior_and_enumeration_remain_compatible(self):
        for model in self.models:
            with self.subTest(model=type(model).__name__):
                expected = model.prior.logpdf(self.gamma) + model.loglik(gamma=self.gamma)
                np.testing.assert_array_equal(model.logpost(theta=self.gamma), expected)
                gamma, logpost = model.complete_enum()
                np.testing.assert_array_equal(gamma, self.gamma)
                np.testing.assert_array_equal(logpost, expected)


if __name__ == "__main__":
    unittest.main()
