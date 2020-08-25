import string

import pandas as pd
import doctest
import warnings

from texthero import visualization, preprocessing, representation
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
    Test visualize_topics
    """

    def test_visualize_topics_clustering_for_second_input(self):

        s = pd.Series(
            [
                "Football, Sports, Soccer",
                "music, violin, orchestra",
                "football, fun, sports",
                "music, band, guitar",
            ]
        )

        s_tfidf = (
            s.pipe(preprocessing.clean)
            .pipe(preprocessing.tokenize)
            .pipe(representation.tfidf)
        )
        s_cluster = (
            s_tfidf.pipe(representation.normalize)
            .pipe(representation.pca, n_components=2)
            .pipe(representation.kmeans, n_clusters=2)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIsNotNone(
                visualization.visualize_topics(s_tfidf, s_cluster)
            )

    def test_visualize_topics_topic_modelling_for_second_input(self):

        s = pd.Series(
            [
                "Football, Sports, Soccer",
                "music, violin, orchestra",
                "football, fun, sports",
                "music, band, guitar",
            ]
        )

        s_tfidf = (
            s.pipe(preprocessing.clean)
            .pipe(preprocessing.tokenize)
            .pipe(representation.tfidf)
        )
        s_lda = s_tfidf.pipe(representation.normalize).pipe(
            representation.lda, n_components=2
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIsNotNone(
                visualization.visualize_topics(s_tfidf, s_lda)
            )

    def test_top_words_per_document(self):
        s = pd.Series(
            [
                "Football, Sports, Soccer",
                "music, violin, orchestra",
                "football, fun, sports",
                "music, band, guitar",
            ]
        )

        s_tfidf = (
            s.pipe(preprocessing.clean)
            .pipe(preprocessing.tokenize)
            .pipe(representation.tfidf)
        )
        s_result = visualization.top_words_per_document(s_tfidf, n_words=2)

        s_true = pd.Series(
            [
                ["soccer", "sports"],
                ["violin", "orchestra"],
                ["fun", "sports"],
                ["guitar", "band"],
            ],
        )
        pd.testing.assert_series_equal(s_result, s_true)

    def test_top_words_per_topic(self):
        s = pd.Series(
            [
                "Football, Sports, Soccer",
                "music, violin, orchestra",
                "football, fun, sports",
                "music, band, guitar",
            ]
        )
        s_tfidf = (
            s.pipe(preprocessing.clean)
            .pipe(preprocessing.tokenize)
            .pipe(representation.tfidf)
        )
        s_cluster = (
            s_tfidf.pipe(representation.normalize)
            .pipe(representation.pca, n_components=2, random_state=42)
            .pipe(representation.kmeans, n_clusters=2, random_state=42)
        )
        s_result = visualization.top_words_per_topic(s_tfidf, s_cluster, n_words=3)
        s_true = pd.Series(
            [["music", "violin", "orchestra"], ["sports", "football", "soccer"]],
        )
        pd.testing.assert_series_equal(s_result, s_true, check_names=False)
