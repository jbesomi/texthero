import pandas as pd
import numpy as np
from flair.embeddings import (
    WordEmbeddings,
    DocumentPoolEmbeddings,
    DocumentRNNEmbeddings,
    TransformerDocumentEmbeddings,
    SentenceTransformerDocumentEmbeddings,
)

from . import PandasTestCase

import doctest
import unittest
import string
import math
import warnings
from parameterized import parameterized

from texthero import embeddings, preprocessing, _types


"""
Test doctest
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(embeddings))
    return tests


s_tokenized = pd.Series(
    [["Test", "Test2", "!", "yes", "hä", "^°"], ["Test3", "wow  ", "aha", "super"]],
    index=[5, 7],
)


"""
Test embeddings functions.
"""


class TestEmbeddings(PandasTestCase):
    """
    Test embed function.

    There are three types of Document Embeddings that
    don't require additional dependencies
    (the SentenceTransformerDocumentEmbeddings requires extra
    dependencies),
    see `here https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md`_.
    We test all of them here.
    """

    def test_embed_document_pool_embedding(self):
        word_embedding = WordEmbeddings("turian")
        document_embedding = DocumentPoolEmbeddings([word_embedding])

        s_return = embeddings.embed(s_tokenized, document_embedding)

        self.assertTrue(isinstance(s_return.iloc[0], list))
        self.assertTrue(isinstance(s_return.iloc[1], list))
        self.assertTrue(len(s_return.iloc[0]) == len(s_return.iloc[1]) > 0)

        pd.testing.assert_index_equal(s_return.index, s_return.index)

        # check if output is valid VectorSeries
        try:
            _types.VectorSeries.check_series(s_return)
        except:
            self.fail("Output is not a valid VectorSeries.")

        del word_embedding

    def test_embed_document_rnn_embedding(self):
        word_embedding = WordEmbeddings("turian")
        document_embedding = DocumentPoolEmbeddings([word_embedding])

        s_return = embeddings.embed(s_tokenized, document_embedding)

        self.assertTrue(isinstance(s_return.iloc[0], list))
        self.assertTrue(isinstance(s_return.iloc[1], list))
        self.assertTrue(len(s_return.iloc[0]) == len(s_return.iloc[1]) > 0)

        pd.testing.assert_index_equal(s_return.index, s_return.index)

        # check if output is valid VectorSeries
        try:
            _types.VectorSeries.check_series(s_return)
        except:
            self.fail("Output is not a valid VectorSeries.")

        del word_embedding

    def test_embed_transformer_document_embedding(self):
        # load smallest available transformer model
        document_embedding = TransformerDocumentEmbeddings(
            "google/reformer-crime-and-punishment"
        )

        s_return = embeddings.embed(s_tokenized, document_embedding)

        self.assertTrue(isinstance(s_return.iloc[0], list))
        self.assertTrue(isinstance(s_return.iloc[1], list))
        self.assertTrue(len(s_return.iloc[0]) == len(s_return.iloc[1]) > 0)

        pd.testing.assert_index_equal(s_return.index, s_return.index)

        # check if output is valid VectorSeries
        try:
            _types.VectorSeries.check_series(s_return)
        except:
            self.fail("Output is not a valid VectorSeries.")

        del document_embedding
