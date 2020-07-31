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
from texthero.representation import flatten

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

    """
    flatten.
    """

    def test_flatten(self):
        index = pd.MultiIndex.from_tuples(
            [("doc0", "Word1"), ("doc0", "Word3"), ("doc1", "Word2")],
            names=["document", "word"],
        )
        s = pd.Series([3, np.nan, 4], index=index)

        s_true = pd.Series(
            [[3.0, 0.0, np.nan], [0.0, 4.0, 0.0]], index=["doc0", "doc1"],
        )

        pd.testing.assert_series_equal(flatten(s), s_true, check_names=False)

    def test_flatten_fill_missing_with(self):
        index = pd.MultiIndex.from_tuples(
            [("doc0", "Word1"), ("doc0", "Word3"), ("doc1", "Word2")],
            names=["document", "word"],
        )
        s = pd.Series([3, np.nan, 4], index=index)

        s_true = pd.Series(
            [[3.0, "FILLED", np.nan], ["FILLED", 4.0, "FILLED"]],
            index=["doc0", "doc1"],
            name="document",
        )

        pd.testing.assert_series_equal(
            flatten(s, fill_missing_with="FILLED"), s_true, check_names=False
        )

    def test_flatten_missing_row(self):
        # Simulating a row with no features, so it's completely missing from
        # the representation series.
        index = pd.MultiIndex.from_tuples(
            [("doc0", "Word1"), ("doc0", "Word3"), ("doc1", "Word2")],
            names=["document", "word"],
        )
        s = pd.Series([3, np.nan, 4], index=index)

        s_true = pd.Series(
            [[3.0, 0.0, np.nan], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
            index=["doc0", "doc1", "doc2"],
            name="document",
        )

        pd.testing.assert_series_equal(
            flatten(s, index=s_true.index), s_true, check_names=False
        )
