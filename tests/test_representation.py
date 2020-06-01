import pandas as pd
from texthero import representation

from . import PandasTestCase

import doctest
import unittest
import string

"""
Test doctest
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(representation))
    return tests


class TestRepresentation(PandasTestCase):
    """
    Term Frequency.
    """

    def test_term_frequency_single_document(self):
        s = pd.Series("a b c c")
        s_true = pd.Series([[1, 1, 2]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_multiple_documents(self):
        s = pd.Series(["doc_one", "doc_two"])
        s_true = pd.Series([[1, 0], [0, 1]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_not_lowercase(self):
        s = pd.Series(["one ONE"])
        s_true = pd.Series([[1, 1]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_punctuation_are_kept(self):
        s = pd.Series(["one !"])
        s_true = pd.Series([[1, 1]])
        self.assertEqual(representation.term_frequency(s), s_true)

    """
    TF-IDF
    """

    def test_idf_single_document(self):
        s = pd.Series("a")
        s_true = pd.Series([[1]])
        self.assertEqual(representation.tfidf(s), s_true)

    def test_idf_single_not_lowercase(self):
        tfidf_single_smooth = 0.7071067811865475  # TODO

        s = pd.Series("ONE one")
        s_true = pd.Series([[tfidf_single_smooth, tfidf_single_smooth]])
        self.assertEqual(representation.tfidf(s), s_true)
