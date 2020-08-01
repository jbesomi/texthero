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
