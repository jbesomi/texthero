import pandas as pd
from texthero import representation
from texthero import preprocessing

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

    """
    Word2Vec
    """

    def test_word2vec(self):
        s = pd.Series(["today is a beautiful day", "today is not that beautiful"])
        df_true = pd.DataFrame(
            [[0.0] * 300] * 7,
            index=["a", "beautiful", "day", "is", "not", "that", "today"],
        )

        s = preprocessing.tokenize(s)

        df_embedding = representation.word2vec(s, min_count=1, seed=1)

        self.assertEqual(type(df_embedding), pd.DataFrame)

        self.assertEqual(df_embedding.shape, df_true.shape)

    def test_most_similar_simple(self):
        s = pd.Series(["one one one"])
        s = preprocessing.tokenize(s)
        df_embeddings = representation.word2vec(s, min_count=1, seed=1)

        to = "one"
        most_similar = representation.most_similar(df_embeddings, to)

        self.assertEqual(most_similar.shape, (1,))

    def test_most_similar_raise_with_series(self):
        s_embed = pd.Series({"one": 1})
        to = "one"

        with self.assertRaisesRegex(ValueError, r"Pandas|pandas"):
            representation.most_similar(s_embed, to)

    def test_most_similar_raise_with_not_in_index(self):
        s_embed = pd.DataFrame(data=[1], index=["one"])
        to = "two"
        with self.assertRaisesRegex(ValueError, r"index"):
            representation.most_similar(s_embed, to)
