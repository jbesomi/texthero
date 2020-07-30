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
        s_true.rename_axis("document", inplace=True)

        self.assertEqual(representation.count(s, return_flat_series=True), s_true)

    def test_count_multiple_documents(self):
        s = pd.Series(["doc_one", "doc_two"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 0], [0, 1]])
        s_true.rename_axis("document", inplace=True)

        self.assertEqual(representation.count(s, return_flat_series=True), s_true)

    def test_count_not_lowercase(self):
        s = pd.Series(["one ONE"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1]])
        s_true.rename_axis("document", inplace=True)

        self.assertEqual(representation.count(s, return_flat_series=True), s_true)

    def test_count_punctuation_are_kept(self):
        s = pd.Series(["one !"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1, 1]])
        s_true.rename_axis("document", inplace=True)

        self.assertEqual(representation.count(s, return_flat_series=True), s_true)

    def test_count_not_tokenized_yet(self):
        s = pd.Series("a b c c")
        s_true = pd.Series([[1, 1, 2]])
        s_true.rename_axis("document", inplace=True)

        with warnings.catch_warnings():  # avoid print warning
            warnings.simplefilter("ignore")
            self.assertEqual(representation.count(s, return_flat_series=True), s_true)

        with self.assertWarns(DeprecationWarning):  # check raise warning
            representation.count(s, return_flat_series=True)

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
        self.assertEqual(representation.tfidf(s, return_flat_series=True), s_true)

    def test_tfidf_single_document(self):
        s = pd.Series("a", index=["yo"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1]], index=["yo"])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s, return_flat_series=True), s_true)

    def test_tfidf_not_tokenized_yet(self):
        s = pd.Series("a")
        s_true = pd.Series([[1]])
        s_true.rename_axis("document", inplace=True)

        with warnings.catch_warnings():  # avoid print warning
            warnings.simplefilter("ignore")
            self.assertEqual(representation.tfidf(s, return_flat_series=True), s_true)

        with self.assertWarns(DeprecationWarning):  # check raise warning
            representation.tfidf(s, return_flat_series=True)

    def test_tfidf_single_not_lowercase(self):
        s = pd.Series("ONE one")
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[1.0, 1.0]])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(representation.tfidf(s, return_flat_series=True), s_true)

    """
    Term Frequency
    """

    def test_term_frequency_single_document(self):
        s = pd.Series("a b c c")
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[0.25, 0.25, 0.5]])
        s_true.rename_axis("document", inplace=True)
        self.assertEqual(
            representation.term_frequency(s, return_flat_series=True), s_true,
        )

    def test_term_frequency_multiple_documents(self):
        s = pd.Series(["doc_one", "doc_two"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[0.5, 0.0], [0.0, 0.5]])
        s_true.rename_axis("document", inplace=True)

        self.assertEqual(
            representation.term_frequency(s, return_flat_series=True), s_true
        )

    def test_term_frequency_not_lowercase(self):
        s = pd.Series(["one ONE"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[0.5, 0.5]])
        s_true.rename_axis("document", inplace=True)

        self.assertEqual(
            representation.term_frequency(s, return_flat_series=True), s_true
        )

    def test_term_frequency_punctuation_are_kept(self):
        s = pd.Series(["one !"])
        s = preprocessing.tokenize(s)
        s_true = pd.Series([[0.5, 0.5]])
        s_true.rename_axis("document", inplace=True)

        self.assertEqual(
            representation.term_frequency(s, return_flat_series=True), s_true
        )

    def test_term_frequency_not_tokenized_yet(self):
        s = pd.Series("a b c c")
        s_true = pd.Series([[0.25, 0.25, 0.5]])
        s_true.rename_axis("document", inplace=True)

        with warnings.catch_warnings():  # avoid print warning
            warnings.simplefilter("ignore")
            self.assertEqual(
                representation.term_frequency(s, return_flat_series=True), s_true
            )

        with self.assertWarns(DeprecationWarning):  # check raise warning
            representation.term_frequency(s, return_flat_series=True)

    """
    Representation series testing
    """

    """
    Count.
    """

    def test_count_single_document_representation_series(self):
        s = pd.Series([list("abbcc")])

        idx = pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c")], names=("document", "word")
        )

        s_true = pd.Series([1, 2, 2], index=idx, dtype="float").astype(
            pd.SparseDtype("int", 0)
        )
        self.assertEqual(representation.count(s), s_true)

    def test_count_multiple_documents_representation_series(self):

        s = pd.Series([["doc_one"], ["doc_two"]])

        idx = pd.MultiIndex.from_tuples(
            [(0, "doc_one"), (1, "doc_two")], names=("document", "word")
        )

        s_true = pd.Series([1, 1], index=idx, dtype="float").astype(
            pd.SparseDtype("int", 0)
        )
        self.assertEqual(representation.count(s), s_true)

    def test_count_not_lowercase_representation_series(self):

        s = pd.Series([["A"], ["a"]])

        idx = pd.MultiIndex.from_tuples(
            [(0, "A"), (1, "a")], names=("document", "word")
        )

        s_true = pd.Series([1, 1], index=idx, dtype="float").astype(
            pd.SparseDtype("int", 0)
        )
        self.assertEqual(representation.count(s), s_true)

    def test_count_punctuation_are_kept_representation_series(self):

        s = pd.Series([["number", "one", "!", "?"]])

        idx = pd.MultiIndex.from_tuples(
            [(0, "!"), (0, "?"), (0, "number"), (0, "one")], names=("document", "word")
        )

        s_true = pd.Series([1, 1, 1, 1], index=idx, dtype="float").astype(
            pd.SparseDtype("int", 0)
        )
        self.assertEqual(representation.count(s), s_true)

    """
    Term Frequency.
    """

    def test_term_frequency_single_document_representation_series(self):
        s = pd.Series([list("abbcc")])

        idx = pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c")], names=("document", "word")
        )

        s_true = pd.Series([0.2, 0.4, 0.4], index=idx, dtype="float").astype(
            pd.SparseDtype("float", np.nan)
        )
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_multiple_documents_representation_series(self):

        s = pd.Series([["doc_one"], ["doc_two"]])

        idx = pd.MultiIndex.from_tuples(
            [(0, "doc_one"), (1, "doc_two")], names=("document", "word")
        )

        s_true = pd.Series([0.5, 0.5], index=idx, dtype="float").astype(
            pd.SparseDtype("float", np.nan)
        )
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_not_lowercase_representation_series(self):

        s = pd.Series([["A"], ["a"]])

        idx = pd.MultiIndex.from_tuples(
            [(0, "A"), (1, "a")], names=("document", "word")
        )

        s_true = pd.Series([0.5, 0.5], index=idx, dtype="float").astype(
            pd.SparseDtype("float", np.nan)
        )
        self.assertEqual(representation.term_frequency(s), s_true)

    def test_term_frequency_punctuation_are_kept_representation_series(self):

        s = pd.Series([["number", "one", "!", "?"]])

        idx = pd.MultiIndex.from_tuples(
            [(0, "!"), (0, "?"), (0, "number"), (0, "one")], names=("document", "word")
        )

        s_true = pd.Series([0.25, 0.25, 0.25, 0.25], index=idx, dtype="float").astype(
            pd.SparseDtype("float", np.nan)
        )
        self.assertEqual(representation.term_frequency(s), s_true)

    """
    TF-IDF
    """

    def test_tfidf_simple_representation_series(self):
        s = pd.Series([["a"]])

        idx = pd.MultiIndex.from_tuples([(0, "a")], names=("document", "word"))
        s_true = pd.Series([1.0], index=idx).astype("Sparse")
        self.assertEqual(representation.tfidf(s), s_true)

    def test_tfidf_single_not_lowercase_representation_series(self):
        tfidf_single_smooth = 1.0

        s = pd.Series([list("Aa")])

        idx = pd.MultiIndex.from_tuples(
            [(0, "A"), (0, "a")], names=("document", "word")
        )

        s_true = pd.Series(
            [tfidf_single_smooth, tfidf_single_smooth], index=idx
        ).astype("Sparse")

        self.assertEqual(representation.tfidf(s), s_true)

    def test_tfidf_single_different_index_representation_series(self):

        idx = pd.MultiIndex.from_tuples(
            [(10, "Bye"), (10, "Hi"), (11, "Bye"), (11, "Test")],
            names=("document", "word"),
        )
        s_true = pd.Series(
            [
                1.0 * (math.log(3 / 3) + 1),
                1.0 * (math.log(3 / 2) + 1),
                2.0 * (math.log(3 / 3) + 1),
                1.0 * (math.log(3 / 2) + 1),
            ],
            index=idx,
        ).astype("Sparse")

        s = pd.Series(["Hi Bye", "Test Bye Bye"], index=[10, 11])
        s = preprocessing.tokenize(s)
        self.assertEqual(representation.tfidf(s), s_true)
