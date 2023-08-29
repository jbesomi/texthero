import pandas as pd
import numpy as np
from texthero import representation
from texthero import preprocessing

import pytest

from .conftest import broken_case

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
    broken_case(
        "count",
        representation.count,
        pd.DataFrame(
            [[1, 0, 0, 1, 2], [0, 2, 1, 0, 1]],
            index=tokenized_output_index,
            columns=["!", ".", "?", "TEST", "Test"],
        ).astype("Sparse[int64, 0]"),
    ),
    broken_case(
        "term_frequency",
        representation.term_frequency,
        pd.DataFrame(
            [[0.25, 0.0, 0.0, 0.25, 0.5], [0.0, 0.5, 0.25, 0.0, 0.25]],
            index=tokenized_output_index,
            columns=["!", ".", "?", "TEST", "Test"],
            dtype="Sparse",
        ).astype("Sparse[float64, nan]"),
    ),
    broken_case(
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
    ),
]


test_cases_vectorization_min_df = [
    # format: [function_name, function, correct output for tokenized input above]
    broken_case(
        "count",
        representation.count,
        pd.DataFrame([2, 1], index=tokenized_output_index, columns=["Test"],).astype(
            "Sparse[int64, 0]"
        ),
    ),
    broken_case(
        "term_frequency",
        representation.term_frequency,
        pd.DataFrame([1, 1], index=tokenized_output_index, columns=["Test"],).astype(
            "Sparse[float64, nan]"
        ),
    ),
    broken_case(
        "tfidf",
        representation.tfidf,
        pd.DataFrame([2, 1], index=tokenized_output_index, columns=["Test"],).astype(
            "Sparse[float64, nan]"
        ),
    ),
]


vector_s = pd.Series([[1.0, 0.0], [0.0, 0.0]], index=[5, 7])
df = pd.DataFrame([[1.0, 0.0], [0.0, 0.0]], index=[5, 7], columns=["a", "b"],).astype(
    "Sparse[float64, nan]"
)


test_cases_dim_reduction_and_clustering = [
    # format: [function_name, function, correct output for s_vector_series and df input above]
    ["pca", representation.pca, pd.Series([[-0.5, 0.0], [0.5, 0.0]], index=[5, 7],),],
    broken_case(
        "nmf",
        representation.nmf,
        pd.Series([[5.119042424626627, 0.0], [0.0, 0.0]], index=[5, 7],),
    ),
    broken_case(
        "tsne",
        representation.tsne,
        pd.Series([[164.86682, 1814.1647], [-164.8667, -1814.1644]], index=[5, 7],),
    ),
    broken_case(
        "kmeans",
        representation.kmeans,
        pd.Series([1, 0], index=[5, 7], dtype="category"),
    ),
    broken_case(
        "dbscan",
        representation.dbscan,
        pd.Series([-1, -1], index=[5, 7], dtype="category"),
    ),
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


class TestAbstractRepresentation:
    """
    Class for representation test cases. Most tests are
    parameterized, some are implemented individually
    (e.g. to test a formula manually).
    """

    """
    Vectorization.
    """

    @pytest.mark.parametrize("name, test_function, correct_output", test_cases_vectorization)
    def test_vectorization_simple(self, name, test_function, correct_output):
        s_true = correct_output
        result_s = test_function(s_tokenized)
        pd.testing.assert_frame_equal(s_true, result_s, check_dtype=False)

    @pytest.mark.parametrize("name, test_function, correct_output", test_cases_vectorization)
    def test_vectorization_noncontinuous_index_kept(
        self, name, test_function, correct_output
    ):
        result_s = test_function(s_tokenized_with_noncontinuous_index)
        pd.testing.assert_index_equal(
            tokenized_output_noncontinous_index, result_s.index
        )

    @pytest.mark.parametrize("name, test_function, correct_output", test_cases_vectorization_min_df)
    def test_vectorization_min_df(self, name, test_function, correct_output):
        s_true = correct_output
        result_s = test_function(s_tokenized, min_df=2)
        pd.testing.assert_frame_equal(s_true, result_s, check_dtype=False)

    @pytest.mark.parametrize("name, test_function, correct_output", test_cases_vectorization)
    def test_vectorization_not_tokenized_yet_warning(self, name, test_function, correct_output):
        with self.assertWarns(DeprecationWarning):  # check raise warning
            test_function(s_not_tokenized)

    @pytest.mark.parametrize("name, test_function, correct_output", test_cases_vectorization)
    def test_vectorization_arguments_to_sklearn(self, name, test_function, correct_output):
        try:
            test_function(s_tokenized, max_features=1, min_df=1, max_df=1.0)
        except TypeError:
            self.fail("Sklearn arguments not handled correctly.")

    """
    Dimensionality Reduction and Clustering
    """

    @pytest.mark.parametrize("name, test_function, correct_output", test_cases_dim_reduction_and_clustering)
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

        # Binary categories: also test if it equals with
        # the category labels inverted (e.g. [0, 1, 0] instead
        # of [1, 0, 1], which makes no difference functionally)
        if pd.api.types.is_categorical_dtype(result_s):
            if len(result_s.cat.categories) == 2 and all(
                result_s.cat.categories == [0, 1]
            ):
                try:
                    result_s_inverted = result_s.apply(lambda category: 1 - category)
                    pd.testing.assert_series_equal(
                        s_true,
                        result_s_inverted,
                        check_dtype=False,
                        rtol=0.1,
                        atol=0.1,
                        check_category_order=False,
                        check_categorical=False,
                    )
                    return
                # inverted comparison fails -> continue to normal comparison
                except AssertionError:
                    pass

        pd.testing.assert_series_equal(
            s_true,
            result_s,
            check_dtype=False,
            rtol=0.1,
            atol=0.1,
            check_category_order=False,
        )

    @pytest.mark.parametrize("name, test_function, correct_output", test_cases_dim_reduction_and_clustering)
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

        # Binary categories: also test if it equals with
        # the category labels inverted (e.g. [0, 1, 0] instead
        # of [1, 0, 1], which makes no difference functionally)
        if pd.api.types.is_categorical_dtype(result_s):
            if len(result_s.cat.categories) == 2 and all(
                result_s.cat.categories == [0, 1]
            ):
                try:
                    result_s_inverted = result_s.apply(lambda category: 1 - category)
                    pd.testing.assert_series_equal(
                        s_true,
                        result_s_inverted,
                        check_dtype=False,
                        rtol=0.1,
                        atol=0.1,
                        check_category_order=False,
                        check_categorical=False,
                    )
                    return
                # inverted comparison fails -> continue to normal comparison
                except AssertionError:
                    pass

        pd.testing.assert_series_equal(
            s_true,
            result_s,
            check_dtype=False,
            rtol=0.1,
            atol=0.1,
            check_category_order=False,
            check_categorical=False,
        )

    @pytest.mark.skip_broken
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
