import pandas as pd
import numpy as np
from texthero import representation
from texthero import preprocessing

from . import PandasTestCase

import doctest
import unittest
import string
import math
import warnings
from parameterized import parameterized


"""
Test doctest
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(representation))
    return tests


"""
Helper functions for the tests.
"""


def _tfidf(term, corpus, document_index):
    idf = (
        math.log((1 + len(corpus)) / (1 + len([doc for doc in corpus if term in doc])))
        + 1
    )
    tfidf_value = idf * corpus[document_index].count(term)
    return tfidf_value


"""
Test functions in representation module in a
parameterized way.
"""


# Define valid inputs / outputs / indexes for different functions.
s_not_tokenized = pd.Series(["This is not tokenized!"])
s_tokenized = pd.Series([["Test", "Test", "TEST", "!"], ["Test", "?", ".", "."]])
s_tokenized_with_noncontinuous_index = pd.Series(
    [["Test", "Test", "TEST", "!"], ["Test", "?", ".", "."]], index=[5, 7]
)

s_tokenized_output_index = [0,1]

s_tokenized_output_index_noncontinous = [5,7]

test_cases_vectorization = [
    # format: [function_name, function, correct output for tokenized input above, dtype of output]
    ["count", representation.count, [1, 1, 2, 2, 1, 1], "int"],
    [
        "term_frequency",
        representation.term_frequency,
        [0.125, 0.125, 0.250, 0.250, 0.125, 0.125],
        "float",
    ],
    [
        "tfidf",
        representation.tfidf,
        [_tfidf(x[1], s_tokenized, x[0]) for x in s_tokenized_output_index],
        "float",
    ],
]

test_cases_vectorization_min_df = [
    # format: [function_name, function, correct output for tokenized input above, dtype of output]
    ["count", representation.count, [2, 1], "int"],
    ["term_frequency", representation.term_frequency, [0.666667, 0.333333], "float",],
    ["tfidf", representation.tfidf, [2.0, 1.0], "float",],
]


class AbstractRepresentationTest(PandasTestCase):
    """
    Class for representation test cases. Most tests are
    parameterized, some are implemented individually
    (e.g. to test a formula manually).
    """

    """
    Vectorization.
    """

    @parameterized.expand(test_cases_vectorization)
    def test_vectorization_simple(
        self, name, test_function, correct_output_values, int_or_float
    ):
        if int_or_float == "int":
            s_true = pd.Series(
                correct_output_values, index=s_tokenized_output_index, dtype="int"
            ).astype(pd.SparseDtype(np.int64, 0))
        else:
            s_true = pd.Series(
                correct_output_values, index=s_tokenized_output_index, dtype="float"
            ).astype(pd.SparseDtype("float", np.nan))
        result_s = test_function(s_tokenized)

        pd.testing.assert_series_equal(s_true, result_s)

    @parameterized.expand(test_cases_vectorization)
    def test_vectorization_noncontinuous_index_kept(
        self, name, test_function, correct_output_values, int_or_float
    ):
        if int_or_float == "int":
            s_true = pd.Series(
                correct_output_values,
                index=s_tokenized_output_noncontinuous_index,
                dtype="int",
            ).astype(pd.SparseDtype(np.int64, 0))
        else:
            s_true = pd.Series(
                correct_output_values,
                index=s_tokenized_output_noncontinuous_index,
                dtype="float",
            ).astype(pd.SparseDtype("float", np.nan))

        result_s = test_function(s_tokenized_with_noncontinuous_index)

        pd.testing.assert_series_equal(s_true, result_s)

    @parameterized.expand(test_cases_vectorization_min_df)
    def test_vectorization_min_df(
        self, name, test_function, correct_output_values, int_or_float
    ):
        if int_or_float == "int":
            s_true = pd.Series(
                correct_output_values,
                index=s_tokenized_output_min_df_index,
                dtype="int",
            ).astype(pd.SparseDtype(np.int64, 0))
        else:
            s_true = pd.Series(
                correct_output_values,
                index=s_tokenized_output_min_df_index,
                dtype="float",
            ).astype(pd.SparseDtype("float", np.nan))

        result_s = test_function(s_tokenized, min_df=2)

        pd.testing.assert_series_equal(s_true, result_s)

    @parameterized.expand(test_cases_vectorization)
    def test_vectorization_not_tokenized_yet_warning(self, name, test_function, *args):
        with self.assertWarns(DeprecationWarning):  # check raise warning
            test_function(s_not_tokenized)

    @parameterized.expand(test_cases_vectorization)
    def test_vectorization_arguments_to_sklearn(self, name, test_function, *args):
        try:
            test_function(s_not_tokenized, max_features=1, min_df=1, max_df=1.0)
        except TypeError:
            self.fail("Sklearn arguments not handled correctly.")

    """
    Individual / special tests.
    """

    def test_tfidf_formula(self):
        s = pd.Series(["Hi Bye", "Test Bye Bye"])
        s = preprocessing.tokenize(s)
        s_true_index = pd.MultiIndex.from_tuples(
            [(0, "Bye"), (0, "Hi"), (1, "Bye"), (1, "Test")],
        )
        s_true = pd.Series(
            [_tfidf(x[1], s, x[0]) for x in s_true_index], index=s_true_index
        ).astype("Sparse")

        self.assertEqual(representation.tfidf(s), s_true)
