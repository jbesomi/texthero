import pandas as pd
import numpy as np
from texthero import nlp

from . import PandasTestCase
import doctest
import unittest
import string

"""
Test doctest
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(nlp))
    return tests


class TestNLP(PandasTestCase):
    """
    Named entity.
    """

    def test_named_entities(self):
        s = pd.Series("New York is a big city")
        s_true = pd.Series([[("New York", "GPE", 0, 8)]])
        self.assertEqual(nlp.named_entities(s), s_true)

    """
    Noun chunks.
    """

    def test_noun_chunks(self):
        s = pd.Series("Today is such a beautiful day")
        s_true = pd.Series(
            [[("Today", "NP", 0, 5), ("such a beautiful day", "NP", 9, 29)]]
        )
        self.assertEqual(nlp.noun_chunks(s), s_true)

    """
    Count sentences.
    """

    def test_count_sentences(self):
        s = pd.Series("I think ... it counts correctly. Doesn't it? Great!")
        s_true = pd.Series(3)
        self.assertEqual(nlp.count_sentences(s), s_true)

    def test_count_sentences_numeric(self):
        s = pd.Series([13.0, 42.0])
        self.assertRaises(TypeError, nlp.count_sentences, s)

    def test_count_sentences_missing_value(self):
        s = pd.Series(["Test.", np.nan])
        self.assertRaises(TypeError, nlp.count_sentences, s)

    def test_count_sentences_index(self):
        s = pd.Series(["Test"], index=[5])
        counted_sentences_s = nlp.count_sentences(s)
        t_same_index = pd.Series([""], index=[5])

        self.assertTrue(counted_sentences_s.index.equals(t_same_index.index))

    def test_count_sentences_wrong_index(self):
        s = pd.Series(["Test", "Test"], index=[5, 6])
        counted_sentences_s = nlp.count_sentences(s)
        t_different_index = pd.Series(["", ""], index=[5, 7])

        self.assertFalse(counted_sentences_s.index.equals(t_different_index.index))

    def test_infer_lang(self):
        s = pd.Series("This is an English text!.")
        s_true = pd.Series([("en", "0.99999")])
        s_result = nlp.infer_lang(s)
        self.assertEqual(s_result, s_true)
