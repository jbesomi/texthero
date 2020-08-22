"""
Unit-tests for the types module.
"""

import pandas as pd
import numpy as np

from . import PandasTestCase
import doctest
import unittest

from texthero import _types

"""
Doctests.
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(_types))
    return tests


class TestTypes(PandasTestCase):
    """
    InputSeries.
    """

    def test_inputseries_function_executes_correctly(self):
        @_types.InputSeries(_types.TextSeries)
        def f(s, t):
            return t

        s = pd.Series("I'm a TextSeries")
        t = "test"
        self.assertEqual(f(s, t), t)

    def test_inputseries_wrong_type(self):
        @_types.InputSeries(_types.TextSeries)
        def f(s):
            pass

        self.assertRaises(TypeError, f, pd.Series([["token", "ized"]]))

    def test_inputseries_correct_type_textseries(self):
        @_types.InputSeries(_types.TextSeries)
        def f(s):
            pass

        try:
            f(pd.Series("I'm a TextSeries"))
        except TypeError:
            self.fail("Failed although input type is correct.")

    def test_inputseries_correct_type_tokenseries(self):
        @_types.InputSeries(_types.TokenSeries)
        def f(s):
            pass

        try:
            f(pd.Series([["token", "ized"]]))
        except TypeError:
            self.fail("Failed although input type is correct.")

    def test_inputseries_correct_type_vectorseries(self):
        @_types.InputSeries(_types.VectorSeries)
        def f(s):
            pass

        try:
            f(pd.Series([[0.0, 1.0]]))
        except TypeError:
            self.fail("Failed although input type is correct.")

    def test_inputseries_correct_type_documentrepresentationseries(self):
        @_types.InputSeries(_types.DocumentTermDF)
        def f(s):
            pass

        try:
            f(
                pd.DataFrame(
                    [[1, 2, 3]],
                    columns=pd.MultiIndex.from_tuples(
                        [("doc1", "word1"), ("doc1", "word2"), ("doc2", "word1")]
                    ),
                    dtype="Sparse",
                )
            )
        except TypeError:
            self.fail("Failed although input type is correct.")

    def test_several_possible_types_correct_type(self):
        @_types.InputSeries([_types.DocumentTermDF, _types.VectorSeries])
        def f(x):
            pass

        try:
            f(
                pd.DataFrame(
                    [[1, 2, 3]],
                    columns=pd.MultiIndex.from_tuples(
                        [("doc1", "word1"), ("doc1", "word2"), ("doc2", "word1")]
                    ),
                    dtype="Sparse",
                )
            )

            f(pd.Series([[1.0, 2.0]]))

        except TypeError:
            self.fail("Failed although input type is correct.")

    def test_several_possible_types_wrong_type(self):
        @_types.InputSeries([_types.DocumentTermDF, _types.VectorSeries])
        def f(x):
            pass

        self.assertRaises(TypeError, f, pd.Series([["token", "ized"]]))
