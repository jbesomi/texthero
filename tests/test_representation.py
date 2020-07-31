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
Test functions in representation module in a
parameterized way.
"""


# Define valid inputs / outputs / indexes for different functions.
s_not_tokenized = pd.Series(["This is not tokenized!"])
s_tokenized = pd.Series([["Test", "Test", "TEST", "!"], ["Test", "?", ".", "."]])
s_tokenized_output_index = pd.MultiIndex.from_tuples(
    [(0, "!"), (0, "TEST"), (0, "Test"), (1, "."), (1, "?"), (1, "Test")],
    names=["document", "word"],
)

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
        [1.405465, 1.405465, 2.000000, 2.810930, 1.405465, 1.000000],
        "float",
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
    def test_vectorization_not_tokenized_yet_warning(self, name, test_function, *args):
        with self.assertWarns(DeprecationWarning):  # check raise warning
            test_function(s_not_tokenized)

    """
    Individual / special tests.
    """

    def test_tfidf_formula(self):
        s = pd.Series(["Hi Bye", "Test Bye Bye"])
        s = preprocessing.tokenize(s)
        s_true_index = pd.MultiIndex.from_tuples(
            [(0, "Bye"), (0, "Hi"), (1, "Bye"), (1, "Test")], names=["document", "word"]
        )
        s_true = pd.Series(
            [
                1.0 * (math.log(3 / 3) + 1),
                1.0 * (math.log(3 / 2) + 1),
                2.0 * (math.log(3 / 3) + 1),
                1.0 * (math.log(3 / 2) + 1),
            ],
            index=s_true_index,
        ).astype("Sparse")

        self.assertEqual(representation.tfidf(s), s_true)
