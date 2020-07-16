import pandas as pd
import numpy as np
from texthero import representation
from texthero import preprocessing

from . import PandasTestCase

import doctest
import unittest
import string
import math

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
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1, 2]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_multiple_documents(self):
        s = pd.Series(["doc_one", "doc_two"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1, 1, 0], [1, 1, 0, 1]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_not_lowercase(self):
        s = pd.Series(["one ONE"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_punctuation_are_kept(self):
        s = pd.Series(["one !"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1]])
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_not_tokenized_yet(self):
        s = pd.Series("a b c c")
        s_true = pd.Series([[1, 1, 2]])
        self.assertEqual(representation.term_frequency(s), s_true)

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
        self.assertEqual(representation.tfidf(s), s_true)

    def test_tfidf_single_not_lowercase(self):
        s = pd.Series("ONE one")
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1.0, 1.0]])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s), s_true)

    def test_tfidf_max_features(self):
        s = pd.Series("one one two")
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[2.0]])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s, max_features=1), s_true)

    def test_tfidf_min_df(self):
        s = pd.Series([["one"], ["one", "two"]])
        s_true = pd.Series([[1.0], [1.0]])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s, min_df=2), s_true)

    def test_tfidf_max_df(self):
        s = pd.Series([["one"], ["one", "two"]])
        s_true = pd.Series([[0.0], [1.4054651081081644]])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s, max_df=1), s_true)

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

    def test_word2vec_not_tokenized_yet(self):
        s = pd.Series(["today is a beautiful day", "today is not that beautiful"])
        df_true = pd.DataFrame(
            [[0.0] * 300] * 7,
            index=["a", "beautiful", "day", "is", "not", "that", "today"],
        )

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
