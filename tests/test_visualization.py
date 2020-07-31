import string

import pandas as pd
import numpy as np
import doctest

from texthero import visualization
from . import PandasTestCase


"""
Test doctest
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(visualization))
    return tests


class TestVisualization(PandasTestCase):
    """
    Test scatterplot.
    """

    def test_scatterplot_dimension_too_high(self):
        s = pd.Series([[1, 2, 3, 4], [1, 2, 3, 4]])
        df = pd.DataFrame(s)
        self.assertRaises(ValueError, visualization.scatterplot, df, col=0)

    def test_scatterplot_dimension_too_low(self):
        s = pd.Series([[1], [1]])
        df = pd.DataFrame(s)
        self.assertRaises(ValueError, visualization.scatterplot, df, col=0)

    def test_scatterplot_return_figure(self):
        s = pd.Series([[1, 2, 3], [1, 2, 3]])
        df = pd.DataFrame(s)
        ret = visualization.scatterplot(df, col=0, return_figure=True)
        self.assertIsNotNone(ret)

    """
    Test top_words.
    """

    def test_top_words(self):
        s = pd.Series("one two two three three three")
        s_true = pd.Series([1, 3, 2], index=["one", "three", "two"])
        self.assertEqual(visualization.top_words(s).sort_index(), s_true)

    def test_top_words_space_char(self):
        s = pd.Series("one \n\t")
        s_true = pd.Series([1], index=["one"])
        self.assertEqual(visualization.top_words(s), s_true)

    def test_top_words_punctuation_between(self):
        s = pd.Series("can't hello-world u.s.a")
        s_true = pd.Series([1, 1, 1], index=["can't", "hello-world", "u.s.a"])
        self.assertEqual(visualization.top_words(s).sort_index(), s_true)

    def test_top_words_remove_external_punctuation(self):
        s = pd.Series("stop. please!")
        s_true = pd.Series([1, 1], index=["please", "stop"])
        self.assertEqual(visualization.top_words(s).sort_index(), s_true)

    def test_top_words_digits(self):
        s = pd.Series("123 hello h1n1")
        s_true = pd.Series([1, 1, 1], index=["123", "h1n1", "hello"])
        self.assertEqual(visualization.top_words(s).sort_index(), s_true)

    def test_top_words_digits_punctuation(self):
        s = pd.Series("123. .321 -h1n1 -cov2")
        s_true = pd.Series([1, 1, 1, 1], index=["123", "321", "cov2", "h1n1"])
        self.assertEqual(visualization.top_words(s).sort_index(), s_true)

    """
    Test worcloud
    """

    def test_wordcloud(self):
        s = pd.Series("one two three")
        self.assertEqual(visualization.wordcloud(s), None)

    """
    Test automated readability index
    """

    def test_automated_readability_index(self):
        s = pd.Series(["New York is a beautiful city.", "Look: New York!", "Wow"])
        s_true = pd.Series([3.0, 6.0, 0.0])
        self.assertEqual(visualization.automated_readability_index(s), s_true)

    def test_automated_readability_index_index(self):
        s = pd.Series(
            ["New York is a beautiful city.", "Look: New York!", "Wow"],
            index=[5, 6, 7],
        )
        self.assertTrue(
            visualization.automated_readability_index(s).index.equals(s.index)
        )

    def test_automated_readability_index_numeric(self):
        s = pd.Series([1.0, 2.0])
        self.assertRaises(TypeError, visualization.automated_readability_index, s)

    def test_automated_readability_index_nan(self):
        s = pd.Series(["Test", np.nan])
        self.assertRaises(TypeError, visualization.automated_readability_index, s)
