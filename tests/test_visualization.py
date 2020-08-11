import string

import pandas as pd
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
