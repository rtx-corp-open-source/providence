# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import numpy as np

from providence import visualization as viz

SIZE = 100


@pytest.fixture(scope="module")
def prediction_df():
    dummy_df = pd.DataFrame.from_dict(
        {"prediction": np.arange(100) + np.random.randint(0, 15, 100), "tte": np.arange(100) + np.random.randint(0, 15, 100),}
    )
    return dummy_df


@pytest.fixture(scope="module")
def dummy_plot(prediction_df):
    return viz.make_error_plot(prediction_df["prediction"], prediction_df["tte"], kind="reg")


class TestMakeErrorPlot:
    def test_title(self, dummy_plot):
        "Test to ensure the title was set correctly"
        want = "Predicted vs. Actual TTE"
        got = dummy_plot.fig.texts[0].get_text()

        assert got == want

    def test_axes_title(self, dummy_plot):
        """This tests is to ensure that our axes are getting set properly, specifically we should be concerned with the x axis"""
        want_x = "Predicted TTE (prediction)"
        want_y = "Actual TTE"

        got_x = dummy_plot.ax_joint.get_xlabel()
        got_y = dummy_plot.ax_joint.get_ylabel()

        assert got_x == want_x
        assert got_y == want_y


class TestImportTools:
    @classmethod
    def local_has_dependency(cls, dependency: str) -> bool:
        """For testing if we have an external dependency"""
        import importlib

        try:
            importlib.import_module(dependency)
            return True
        except:
            return False

    def test_the_import_test(self):
        # practice environment hygiene
        assert self.local_has_dependency("vega") == False, "Shouldn't be able to import vega. Not a providence dep so this shoul be False"

        # dependency of the project
        assert self.local_has_dependency("matplotlib.pyplot") == True, "Should have matplotlib on local"

        # optional dependency of the project
        assert self.local_has_dependency("seaborn") == True, "Should have seaborn on local"

    def test_checked_import__always_return_good_context_manager(self):
        mpl, ctxmgr, sns = viz.check_for_mpl()
        with ctxmgr:
            assert True, "Context manager should be enterable"
        assert True, "Context manager should exit without complication"

    def test_checked_import__context_manager_does_not_supress_exceptions(self):
        _, ctxmgr, sns = viz.check_for_mpl()
        with pytest.raises(BaseException, match=".* specific type .*") as raise_check:
            with ctxmgr:
                raise ValueError("Testing a specific type of exception, arbitrarily chosen")

    def test_all_dependencies_exist(self):
        deps = viz.check_for_mpl()
        assert all(map(lambda x: x is not None, deps)), "Everything should be loaded successfully, without surprise Nones"

