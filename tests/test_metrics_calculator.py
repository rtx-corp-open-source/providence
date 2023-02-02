"""
**Raytheon Technologies proprietary**
Export controlled - see license file
"""

from pytest import fixture
from providence.distributions import Weibull

from providence.metrics import MetricsCalculator


class TestMetricsCalculator:

    @fixture
    def simple_weibull_metrics_calculator(self, simple_weibull_model, simple_providence_ds) -> MetricsCalculator:
        return MetricsCalculator(simple_weibull_model, Weibull, simple_providence_ds)

    def assert_has_baseline_keys(self, output_df):
        for error_key in [f"error_{stat}" for stat in 'median mean mode'.split(' ')]:
            assert error_key in output_df

        for overshoot_key in [f"{stat}_overshoot" for stat in 'median mean mode'.split(' ')]:
            assert overshoot_key in output_df


    def test_error_by_timestep__works(self, simple_weibull_metrics_calculator: MetricsCalculator):
        output_df = simple_weibull_metrics_calculator.error_by_timestep(max_timestep=5, min_timestep=2)
        self.assert_has_baseline_keys(output_df)

    def test_metrics_by_timestep__works(self, simple_weibull_metrics_calculator: MetricsCalculator):
        output_df = simple_weibull_metrics_calculator.metrics_by_timestep(max_timestep=5, min_timestep=2)
        for key in 'tte mse mfe'.split(' '):
            assert key in output_df, "Missing timestep-metric key"

    def test_percent_overshot_by_tte__works(self, simple_weibull_metrics_calculator: MetricsCalculator):
        output_df = simple_weibull_metrics_calculator.percent_overshot_by_tte(max_timestep=5, min_timestep=2)
        for key in 'mean median mode'.split(' '):
            assert f"%_Overshot_{key.capitalize()}" in output_df, "Missing overshot-metric key"
        
        assert "TTE" in output_df, "TTE column is missing"


