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
from parameterized import parameterized


"""
Test doctest
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(representation))
    return tests


"""
Helper functions for the tests.
"""


def _tfidf(term, corpus, document_index):
    idf = (
        math.log((1 + len(corpus)) / (1 + len([doc for doc in corpus if term in doc])))
        + 1
    )
    tfidf_value = idf * corpus[document_index].count(term)
    return tfidf_value


"""
Test functions in representation module in a
parameterized way.
"""


# Define valid inputs / outputs / indexes for different functions.
s_not_tokenized = pd.Series(["This is not tokenized!"])
s_tokenized = pd.Series([["Test", "Test", "TEST", "!"], ["Test", "?", ".", "."]])
s_tokenized_with_noncontinuous_index = pd.Series(
    [["Test", "Test", "TEST", "!"], ["Test", "?", ".", "."]], index=[5, 7]
)

tokenized_output_index = pd.Index([0, 1])

tokenized_output_noncontinous_index = pd.Index([5, 7])

test_cases_vectorization = [
    # format: [function_name, function, correct output for tokenized input above]
    [
        "count",
        representation.count,
        pd.DataFrame(
            [[1, 0, 0, 1, 2], [0, 2, 1, 0, 1]],
            index=tokenized_output_index,
            columns=["!", ".", "?", "TEST", "Test"],
        ).astype("Sparse[int64, 0]"),
    ],
    [
        "term_frequency",
        representation.term_frequency,
        pd.DataFrame(
            [[0.125, 0.0, 0.0, 0.125, 0.250], [0.0, 0.25, 0.125, 0.0, 0.125]],
            index=tokenized_output_index,
            columns=["!", ".", "?", "TEST", "Test"],
            dtype="Sparse",
        ).astype("Sparse[float64, nan]"),
    ],
    [
        "tfidf",
        representation.tfidf,
        pd.DataFrame(
            [
                [
                    _tfidf(x, s_tokenized, 0)  # Testing the tfidf formula here
                    for x in ["!", ".", "?", "TEST", "Test"]
                ],
                [_tfidf(x, s_tokenized, 1) for x in ["!", ".", "?", "TEST", "Test"]],
            ],
            index=tokenized_output_index,
            columns=["!", ".", "?", "TEST", "Test"],
        ).astype("Sparse[float64, nan]"),
    ],
]


test_cases_vectorization_min_df = [
    # format: [function_name, function, correct output for tokenized input above]
    [
        "count",
        representation.count,
        pd.DataFrame([2, 1], index=tokenized_output_index, columns=["Test"],).astype(
            "Sparse[int64, 0]"
        ),
    ],
    [
        "term_frequency",
        representation.term_frequency,
        pd.DataFrame(
            [0.666667, 0.333333], index=tokenized_output_index, columns=["Test"],
        ).astype("Sparse[float64, nan]"),
    ],
    [
        "tfidf",
        representation.tfidf,
        pd.DataFrame([2, 1], index=tokenized_output_index, columns=["Test"],).astype(
            "Sparse[float64, nan]"
        ),
    ],
]


vector_s = pd.Series([[1.0, 0.0], [0.0, 0.0]], index=[5, 7])
df = pd.DataFrame([[1.0, 0.0], [0.0, 0.0]], index=[5, 7], columns=["a", "b"],).astype(
    "Sparse[float64, nan]"
)


test_cases_dim_reduction_and_clustering = [
    # format: [function_name, function, correct output for s_vector_series and df input above]
    ["pca", representation.pca, pd.Series([[-0.5, 0.0], [0.5, 0.0]], index=[5, 7],),],
    [
        "nmf",
        representation.nmf,
        pd.Series([[5.119042424626627, 0.0], [0.0, 0.0]], index=[5, 7],),
    ],
    [
        "tsne",
        representation.tsne,
        pd.Series([[164.86682, 1814.1647], [-164.8667, -1814.1644]], index=[5, 7],),
    ],
    [
        "kmeans",
        representation.kmeans,
        pd.Series([1, 0], index=[5, 7], dtype="category"),
    ],
    [
        "dbscan",
        representation.dbscan,
        pd.Series([-1, -1], index=[5, 7], dtype="category"),
    ],
    [
        "meanshift",
        representation.meanshift,
        pd.Series([0, 1], index=[5, 7], dtype="category"),
    ],
    [
        "normalize",
        representation.normalize,
        pd.Series([[1.0, 0.0], [0.0, 0.0]], index=[5, 7],),
    ],
]


class AbstractRepresentationTest(PandasTestCase):
    """
    Class for representation test cases. Most tests are
    parameterized, some are implemented individually
    (e.g. to test a formula manually).
    """

    """
    Vectorization.
    """

    @parameterized.expand(test_cases_vectorization)
    def test_vectorization_simple(self, name, test_function, correct_output):
        s_true = correct_output
        result_s = test_function(s_tokenized)
        pd.testing.assert_frame_equal(s_true, result_s, check_dtype=False)

    @parameterized.expand(test_cases_vectorization)
    def test_vectorization_noncontinuous_index_kept(
        self, name, test_function, correct_output=None
    ):
        result_s = test_function(s_tokenized_with_noncontinuous_index)
        pd.testing.assert_index_equal(
            tokenized_output_noncontinous_index, result_s.index
        )

    @parameterized.expand(test_cases_vectorization_min_df)
    def test_vectorization_min_df(self, name, test_function, correct_output):
        s_true = correct_output
        result_s = test_function(s_tokenized, min_df=2)
        pd.testing.assert_frame_equal(s_true, result_s, check_dtype=False)

    @parameterized.expand(test_cases_vectorization)
    def test_vectorization_not_tokenized_yet_warning(self, name, test_function, *args):
        with self.assertWarns(DeprecationWarning):  # check raise warning
            test_function(s_not_tokenized)

    @parameterized.expand(test_cases_vectorization)
    def test_vectorization_arguments_to_sklearn(self, name, test_function, *args):
        try:
            test_function(s_tokenized, max_features=1, min_df=1, max_df=1.0)
        except TypeError:
            self.fail("Sklearn arguments not handled correctly.")

    """
    Dimensionality Reduction and Clustering
    """

    @parameterized.expand(test_cases_dim_reduction_and_clustering)
    def test_dim_reduction_and_clustering_with_vector_series_input(
        self, name, test_function, correct_output
    ):
        s_true = correct_output

        if name == "kmeans":
            result_s = test_function(vector_s, random_state=42, n_clusters=2)
        elif name == "dbscan" or name == "meanshift" or name == "normalize":
            result_s = test_function(vector_s)
        else:
            result_s = test_function(vector_s, random_state=42)

        pd.testing.assert_series_equal(
            s_true,
            result_s,
            check_dtype=False,
            rtol=0.1,
            atol=0.1,
            check_category_order=False,
        )

    @parameterized.expand(test_cases_dim_reduction_and_clustering)
    def test_dim_reduction_and_clustering_with_dataframe_input(
        self, name, test_function, correct_output
    ):
        s_true = correct_output

        if name == "normalize":
            # testing this below separately
            return

        if name == "kmeans":
            result_s = test_function(df, random_state=42, n_clusters=2)
        elif name == "dbscan" or name == "meanshift" or name == "normalize":
            result_s = test_function(df)
        else:
            result_s = test_function(df, random_state=42)

        pd.testing.assert_series_equal(
            s_true,
            result_s,
            check_dtype=False,
            rtol=0.1,
            atol=0.1,
            check_category_order=False,
        )

    def test_normalize_DataFrame_also_as_output(self):
        # normalize should also return DataFrame output for DataFrame
        # input so we test it separately
        result = representation.normalize(df)
        correct_output = pd.DataFrame(
            [[1.0, 0.0], [0.0, 0.0]], index=[5, 7], columns=["a", "b"],
        )

        pd.testing.assert_frame_equal(
            result, correct_output, check_dtype=False, rtol=0.1, atol=0.1,
        )

    """
    Test Topic Modelling (not all are suitable for parameterization).
    `topics_from_topic_model, lda, truncatedSVD` already tested above.

    Here, we test
    `relevant_words_per_document, relevant_words_per_topic, topic_matrices`
    """

    def test_relevant_words_per_document(self):
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
        s_result = representation.relevant_words_per_document(s_tfidf, n_words=2)

        s_true = pd.Series(
            [
                ["soccer", "sports"],
                ["violin", "orchestra"],
                ["fun", "sports"],
                ["guitar", "band"],
            ],
        )
        pd.testing.assert_series_equal(s_result, s_true)

    def test_relevant_words_per_topic(self):
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

        s_document_topic, s_topic_term = representation.topic_matrices(
            s_tfidf, s_cluster
        )
        s_document_topic_distribution = representation.normalize(
            s_document_topic, norm="l1"
        )
        s_topic_term_distribution = representation.normalize(s_topic_term, norm="l1")

        s_result = representation.relevant_words_per_topic(
            s_tfidf, s_document_topic_distribution, s_topic_term_distribution, n_words=3
        )
        s_true = pd.Series(
            [["music", "violin", "orchestra"], ["sports", "football", "soccer"]],
        )
        pd.testing.assert_series_equal(s_result, s_true, check_names=False)

    def test_topic_matrices_clustering_for_second_input(self):

        s = pd.Series(["Football", "Music", "Football", "Music",])

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

        s_document_topic_result, s_topic_term_result = representation.topic_matrices(
            s_tfidf, s_cluster
        )

        s_document_topic_true = pd.DataFrame(
            [[0, 1], [1, 0], [0, 1], [1, 0]],
            columns=pd.MultiIndex.from_tuples(
                [("Document Topic Matrix", 0), ("Document Topic Matrix", 1)]
            ),
        )

        s_topic_term_true = pd.DataFrame(
            [[0.0, 3.021651], [3.021651, 0.0]],
            columns=pd.MultiIndex.from_tuples(
                [("Topic Term Matrix", "football"), ("Topic Term Matrix", "music")]
            ),
        )

        pd.testing.assert_frame_equal(
            s_document_topic_result,
            s_document_topic_true,
            check_less_precise=True,
            check_dtype=False,
        )

        pd.testing.assert_frame_equal(
            s_topic_term_result,
            s_topic_term_true,
            check_less_precise=True,
            check_dtype=False,
        )

    def test_visualize_topics_topic_modelling_for_second_input(self):

        s = pd.Series(["Football", "Music", "Football", "Music",])

        s_tfidf = (
            s.pipe(preprocessing.clean)
            .pipe(preprocessing.tokenize)
            .pipe(representation.tfidf)
        )
        s_lda = s_tfidf.pipe(representation.normalize).pipe(
            representation.lda, n_components=2, random_state=42
        )

        s_document_topic_result, s_topic_term_result = representation.topic_matrices(
            s_tfidf, s_lda
        )

        s_document_topic_true = pd.DataFrame(
            [
                [0.744417, 0.255583],
                [0.255583, 0.744417],
                [0.744417, 0.255583],
                [0.255583, 0.744417],
            ],
            columns=pd.MultiIndex.from_tuples(
                [("Document Topic Matrix", 0), ("Document Topic Matrix", 1)]
            ),
        )

        s_topic_term_true = pd.DataFrame(
            [[2.249368, 0.772283], [0.772283, 2.249369]],
            columns=pd.MultiIndex.from_tuples(
                [("Topic Term Matrix", "football"), ("Topic Term Matrix", "music")]
            ),
        )

        pd.testing.assert_frame_equal(
            s_document_topic_result,
            s_document_topic_true,
            check_less_precise=True,
            check_dtype=False,
        )

        pd.testing.assert_frame_equal(
            s_topic_term_result,
            s_topic_term_true,
            check_less_precise=True,
            check_dtype=False,
        )
