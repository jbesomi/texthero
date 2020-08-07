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

    """
    InputSeries.
    """

    def test_inputseries_function_executes_correctly(self):
        @_helper.InputSeries(_helper.TextSeries)
        def f(s, t):
            return t

        s = pd.Series("I'm a TextSeries")
        t = "test"
        self.assertEqual(f(s, t), t)

    def test_inputseries_wrong_type(self):
        @_helper.InputSeries(_helper.TextSeries)
        def f(s):
            pass

        self.assertRaises(TypeError, f, pd.Series([["token", "ized"]]))

    def test_inputseries_correct_type_textseries(self):
        @_helper.InputSeries(_helper.TextSeries)
        def f(s):
            pass

        try:
            f(pd.Series("I'm a TextSeries"))
        except TypeError:
            self.fail("Failed although input type is correct.")

    def test_inputseries_correct_type_tokenseries(self):
        @_helper.InputSeries(_helper.TokenSeries)
        def f(s):
            pass

        try:
            f(pd.Series([["token", "ized"]]))
        except TypeError:
            self.fail("Failed although input type is correct.")

    def test_inputseries_correct_type_vectorseries(self):
        @_helper.InputSeries(_helper.VectorSeries)
        def f(s):
            pass

        try:
            f(pd.Series([[0.0, 1.0]]))
        except TypeError:
            self.fail("Failed although input type is correct.")

    def test_inputseries_correct_type_documentrepresentationseries(self):
        @_helper.InputSeries(_helper.RepresentationSeries)
        def f(s):
            pass

        try:
            f(
                pd.Series(
                    [1, 2, 3],
                    index=pd.MultiIndex.from_tuples(
                        [("doc1", "word1"), ("doc1", "word2"), ("doc2", "word1")]
                    ),
                )
            )
        except TypeError:
            self.fail("Failed although input type is correct.")
