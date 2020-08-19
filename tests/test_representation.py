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

s_tokenized_output_index = [0, 1]

s_tokenized_output_index_noncontinous = [5, 7]


def _get_multiindex_for_tokenized_output(first_level_name):
    return pd.MultiIndex.from_product(
        [[first_level_name], ["!", ".", "?", "TEST", "Test"]]
    )


test_cases_vectorization = [
    # format: [function_name, function, correct output for tokenized input above]
    [
        "count",
        representation.count,
        pd.DataFrame(
            [[1, 0, 0, 1, 2], [0, 2, 1, 0, 1]],
            index=s_tokenized_output_index,
            columns=_get_multiindex_for_tokenized_output("count"),
        ).astype("Sparse[int64, 0]"),
    ],
    [
        "term_frequency",
        representation.term_frequency,
        pd.DataFrame(
            [[0.125, 0.0, 0.0, 0.125, 0.250], [0.0, 0.25, 0.125, 0.0, 0.125]],
            index=s_tokenized_output_index,
            columns=_get_multiindex_for_tokenized_output("term_frequency"),
        ).astype("Sparse"),
    ],
    [
        "tfidf",
        representation.tfidf,
        pd.DataFrame(
            [
                [
                    _tfidf(x, s_tokenized, 0)  # Testing the tfidf formula here
                    for x in ["!", ".", "?", "TEST", "Test"]
                ],
                [_tfidf(x, s_tokenized, 0) for x in ["!", ".", "?", "TEST", "Test"]],
            ],
            index=s_tokenized_output_index,
            columns=_get_multiindex_for_tokenized_output("tfidf"),
        ).astype("Sparse"),
    ],
]


test_cases_vectorization_min_df = [
    # format: [function_name, function, correct output for tokenized input above]
    [
        "count",
        representation.count,
        pd.DataFrame(
            [2, 1],
            index=s_tokenized_output_index,
            columns=pd.MultiIndex.from_tuples([("count", "Test")]),
        ).astype("Sparse[int64, 0]"),
    ],
    [
        "term_frequency",
        representation.term_frequency,
        pd.DataFrame(
            [0.666667, 0.333333],
            index=s_tokenized_output_index,
            columns=pd.MultiIndex.from_tuples([("term_frequency", "Test")]),
        ).astype("Sparse"),
    ],
    [
        "tfidf",
        representation.tfidf,
        pd.DataFrame(
            [2, 1],
            index=s_tokenized_output_index,
            columns=pd.MultiIndex.from_tuples([("tfidf", "Test")]),
        ).astype("Sparse"),
    ],
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
    def test_vectorization_simple(self, name, test_function, correct_output):
        s_true = correct_output
        result_s = test_function(s_tokenized)
        pd.testing.assert_frame_equal(s_true, result_s, check_less_precise=True, check_dtype = False)

    @parameterized.expand(test_cases_vectorization)
    def test_vectorization_noncontinuous_index_kept(
        self, name, test_function, correct_output=None
    ):
        result_s = test_function(s_tokenized_with_noncontinuous_index)
        pd.testing.assert_frame_equal(s_tokenized_output_index_noncontinous, result_s.index, check_dtype = False)

    @parameterized.expand(test_cases_vectorization_min_df)
    def test_vectorization_min_df(self, name, test_function, correct_output):
        s_true = correct_output
        result_s = test_function(s_tokenized, min_df=2)
        pd.testing.assert_frame_equal(s_true, result_s, check_less_precise=True, check_dtype = False)

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
