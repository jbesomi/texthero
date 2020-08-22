"""
Unit-tests for the helper module.
"""

import pandas as pd
import numpy as np

from . import PandasTestCase
import doctest
import unittest
import warnings

from texthero import _helper

"""
Doctests.
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(_helper))
    return tests


"""
Test Decorators.
"""


class TestHelpers(PandasTestCase):
    """
    handle_nans.
    """

    def test_handle_nans(self):
        s = pd.Series(["Test", np.nan, pd.NA])

        @_helper.handle_nans(replace_nans_with="This was a NAN")
        def f(s):
            return s

        s_true = pd.Series(["Test", "This was a NAN", "This was a NAN"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(f(s), s_true)

        with self.assertWarns(Warning):
            f(s)

    def test_handle_nans_no_nans_in_input(self):
        s = pd.Series(["Test"])

        @_helper.handle_nans(replace_nans_with="This was a NAN")
        def f(s):
            return s

        s_true = pd.Series(["Test"])

        self.assertEqual(f(s), s_true)

    # This is not in test_indexes.py as it requires a custom test case.
    def test_handle_nans_index(self):
        s = pd.Series(["Test", np.nan, pd.NA], index=[4, 5, 6])

        @_helper.handle_nans(replace_nans_with="This was a NAN")
        def f(s):
            return s

        s_true = pd.Series(
            ["Test", "This was a NAN", "This was a NAN"], index=[4, 5, 6]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertTrue(f(s).index.equals(s_true.index))

    def test_pandas_set_item_normal(self):
        df1 = pd.DataFrame([[1, 2], [5, 3]], columns=["Test", "Test2"])
        df2 = pd.DataFrame([0, 1])

        df1["here"] = df2

        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame([[1, 2, 0], [5, 3, 1]], columns=["Test", "Test2", "here"]),
        )

    def test_pandas_set_item_multiIndex(self):
        df1 = pd.DataFrame(["Text 1", "Text 2"], columns=["Test"])
        df2 = pd.DataFrame(
            [[3, 5], [8, 4]], columns=pd.MultiIndex.from_product([["count"], [0, 1]]),
        )

        df1["here"] = df2

        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                [["Text 1", 3, 5], ["Text 2", 8, 4]],
                columns=pd.MultiIndex.from_tuples(
                    [("Test", ""), ("here", 0), ("here", 1)]
                ),
            ),
        )
