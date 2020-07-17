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

"""
Test doctest
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(representation))
    return tests


class TestRepresentation(PandasTestCase):
    """
    Count.
    """

    def test_count_single_document(self):
        s = pd.Series("a b c c")
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1, 2]])
        self.assertEqual(representation.count(s), s_true)

    def test_count_multiple_documents(self):
        s = pd.Series(["doc_one", "doc_two"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1, 1, 0], [1, 1, 0, 1]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_count_not_lowercase(self):
        s = pd.Series(["one ONE"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1]])
        self.assertEqual(representation.count(s), s_true)

    def test_count_punctuation_are_kept(self):
        s = pd.Series(["one !"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1]])
        self.assertEqual(representation.count(s), s_true)

    def test_count_not_tokenized_yet(self):
        s = pd.Series("a b c c")
        s_true = pd.Series([[1, 1, 2]])

        with warnings.catch_warnings():  # avoid print warning
            warnings.simplefilter("ignore")
            self.assertEqual(representation.count(s), s_true)

        with self.assertWarns(DeprecationWarning):  # check raise warning
            representation.term_frequency(s)

    """
    TF-IDF
    """

    def test_tfidf_formula(self):
        s = pd.Series(["Hi Bye", "Test Bye Bye"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series(
            [
                [
                    1.0 * (math.log(3 / 3) + 1),
                    1.0 * (math.log(3 / 2) + 1),
                    0.0 * (math.log(3 / 2) + 1),
                ],
                [
                    2.0 * (math.log(3 / 3) + 1),
                    0.0 * (math.log(3 / 2) + 1),
                    1.0 * (math.log(3 / 2) + 1),
                ],
            ]
        )
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s), s_true)

    def test_tfidf_single_document(self):
        s = pd.Series("a", index=["yo"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1]], index=["yo"])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s), s_true)

    def test_tfidf_not_tokenized_yet(self):
        s = pd.Series("a")
        s_true = pd.Series([[1]])
        s_true.rename_axis("document", inplace=True)

        with warnings.catch_warnings():  # avoid print warning
            warnings.simplefilter("ignore")
            self.assertEqual(representation.tfidf(s), s_true)

        with self.assertWarns(DeprecationWarning):  # check raise warning
            representation.tfidf(s)

    def test_tfidf_single_not_lowercase(self):
        s = pd.Series("ONE one")
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1.0, 1.0]])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s), s_true)

    """
    Term Frequency
    """

    def test_term_frequency_single_document(self):
        s = pd.Series("a b c c")
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[0.25, 0.25, 0.5]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_multiple_documents(self):
        s = pd.Series(["doc_one", "doc_two"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.0], [0.16666666666666666, 0.16666666666666666, 0.0, 0.16666666666666666]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_not_lowercase(self):
        s = pd.Series(["one ONE"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[0.5, 0.5]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_punctuation_are_kept(self):
        s = pd.Series(["one !"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[0.5, 0.5]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_not_tokenized_yet(self):
        s = pd.Series("a b c c")
        s_true = pd.Series([[0.25, 0.25, 0.5]])

        with warnings.catch_warnings():  # avoid print warning
            warnings.simplefilter("ignore")
            self.assertEqual(representation.term_frequency(s), s_true)

        with self.assertWarns(DeprecationWarning):  # check raise warning
            representation.term_frequency(s)
