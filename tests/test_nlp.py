import pandas as pd
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
