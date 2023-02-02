"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""
from numpy import array, isnan, ndarray, NaN
from numpy.testing import assert_almost_equal
from pytest import fixture

from providence.distributions import Weibull
from providence.metrics import fleet_metrics, mse, output_per_device, rmse, score_phm08, smpe


class TestMetrics:
    ################################################################################
    # Helpers
    ################################################################################
    @classmethod
    def functional_set(cls, L: list, index: int, new_val) -> list:
        new_L = L[:]
        new_L[index] = new_val
        return new_L

    ################################################################################
    # Fixtures
    ################################################################################

    @fixture
    def basic_example(self) -> ndarray:
        return array([1, 2, 3, 4, 5])

    ################################################################################
    # Tests: MSE
    ################################################################################

    def test_mean_squared_error__zero(self, basic_example: ndarray):
        assert mse(basic_example, basic_example) == 0.

    def test_mean_squared_error__all_nans(self, basic_example: ndarray):
        assert isnan(mse(basic_example, [NaN] * len(basic_example)))

    def test_mean_squared_error__all_nans_ignored(self, basic_example: ndarray):
        # should emit something like "RuntimeWarning: Mean of empty slice"
        # you're welcome to try to catch that in the logging capture of pytest
        assert isnan(mse(basic_example, [NaN] * len(basic_example), ignore_nans=True))

    def test_mean_squared_error__one_nan_tripped(self, basic_example: ndarray):
        # This (computed) 3 is a magic number within the bounds of the basic_example
        # feel free to make this a bounded random number from a RNG
        nandex = round(len(basic_example) / 2)
        assert isnan(mse(basic_example, self.functional_set(basic_example.tolist(), nandex, NaN)))

    def test_mean_squared_error__one_nan_ignored(self, basic_example: ndarray):
        test_pred = self.functional_set(basic_example.tolist(), 4, NaN)
        assert mse(basic_example, test_pred, ignore_nans=True) == 0

    def test_mean_squared_error__one_nan_ignored__nonzero_error(self, basic_example: ndarray):
        test_pred = [4, 3, 2, 1, NaN]
        # y_true - y_pred == [4 - 1, 3 - 2, 2 - 3, 1 - 4], therefore mse = 1
        assert mse(basic_example, test_pred[::-1], ignore_nans=True) == 1

    ################################################################################
    # Tests: RMSE
    ################################################################################

    def test_root_mean_squared_error__zero(self, basic_example: ndarray):
        assert rmse(basic_example, basic_example) == 0.

    def test_root_mean_squared_error__all_nans(self, basic_example: ndarray):
        assert isnan(rmse(basic_example, [NaN] * len(basic_example)))

    def test_root_mean_squared_error__all_nans_ignored(self, basic_example: ndarray):
        # should emit something like "RuntimeWarning: Mean of empty slice"
        # you're welcome to try to catch that in the logging capture of pytest
        assert isnan(rmse(basic_example, [NaN] * len(basic_example), ignore_nans=True))

    def test_root_mean_squared_error__one_nan_tripped(self, basic_example: ndarray):
        # This (computed) 3 is a magic number within the bounds of the basic_example
        # feel free to make this a bounded random number from a RNG
        nandex = round(len(basic_example) / 2)
        assert isnan(rmse(basic_example, self.functional_set(basic_example.tolist(), nandex, NaN)))

    def test_root_mean_squared_error__one_nan_ignored(self, basic_example: ndarray):
        test_pred = self.functional_set(basic_example.tolist(), 4, NaN)
        assert rmse(basic_example, test_pred, ignore_nans=True) == 0

    def test_root_mean_squared_error__one_nan_ignored__nonzero_error(self, basic_example: ndarray):
        test_pred = [4, 3, 2, 1, NaN]
        # y_true - y_pred == [4 - 1, 3 - 2, 2 - 3, 1 - 4], therefore rmse = 1
        assert rmse(basic_example, test_pred[::-1], ignore_nans=True) == 1

    ################################################################################
    # Tests: SMPE
    ################################################################################

    def test_smpe__basic_math(self):
        "A computation you should readily be able to hand-verify"
        assert_almost_equal(smpe([1, 2], [2, 3]), 4 / 15)

    def test_smpe__zero(self, basic_example: ndarray):
        "This test outlines the zero-cases for smpe: identical input, reversed input, and unhandled nans"
        # language note: abstract / linear algebra notion of kernel as in "that which sends the input to zero"
        for kernel in [basic_example, basic_example[::-1], [NaN] * len(basic_example)]:
            result = smpe(basic_example, kernel)
            assert_almost_equal(result, 0)

    def test_smpe__one(self, basic_example: ndarray):
        "Demonstrates overshooting. Positive = 'up' and 'over'"
        assert smpe([-1] * len(basic_example), basic_example) == 1

    def test_smpe__neg_one(self, basic_example: ndarray):
        "Demonstrates undershooting. Negative = 'down' and 'under'"
        assert smpe(basic_example, [-1] * len(basic_example)) == -1

    def test_smpe__nan_except_one(self, basic_example: ndarray):
        "smpe(([1, 5, 2], [NaN, 2, NaN], ignore_nans = True) ==  1 / 3 * (2 - 5) == (2-5)/(5 + 7)"
        assert_almost_equal(smpe([1, 5, 2], [NaN, 2, NaN], ignore_nans=True), -3 / 7)

    ################################################################################
    # Tests: Score
    ################################################################################

    def test_score_phm08__works(self, basic_example: ndarray):
        assert_almost_equal(score_phm08(basic_example, [1] * len(basic_example)), 0.8661, decimal=4)

    ################################################################################
    # tests: Fleet Metrics
    ################################################################################

    def test_fleet_metrics__works(self, simple_weibull_model, simple_providence_ds):
        df = fleet_metrics(simple_weibull_model, Weibull, simple_providence_ds)
        assert df.shape[0], "Should result in the fleet metrics being computed"

################################################################################
# Tests: metrics module, proper
################################################################################
def test_output_per_device__weibull__works(simple_weibull_model, simple_providence_ds):
    outputs = output_per_device(simple_weibull_model, Weibull, simple_providence_ds)
    must_have_columns = 'alpha beta tte censor mean median mode'.split()
    for col in must_have_columns:
        assert col in outputs.columns, f"{col=} missing from output_per_device() output for Weibull distribution"
