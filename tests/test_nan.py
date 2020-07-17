import pandas as pd
import numpy as np
from texthero import nlp, visualization, preprocessing, representation

from . import PandasTestCase
import unittest
import string
from parameterized import parameterized

# Define valid inputs for different functions.
s_nan = pd.Series(["test1", np.NaN, "test2", pd.NA])
s_numeric_and_nan_lists = pd.Series([[5.0, 5.0], [6.0, 6.0], np.nan])

# Define all test cases. Every test case is a list
# of [name of test case, function to test, tuple of valid input for the function].
# First argument of valid input has to be the Pandas Series we
# want to test with at least one np.nan. If this is different for a function, a separate
# test case has to implemented in the class below.
# The tests will be run by AbstractNaNTest below through the @parameterized
# decorator.
# The names will be expanded automatically, so e.g. "named_entities"
# creates test cases test_ignores_nan_named_entities.

test_cases_nlp = [
    ["named_entities", nlp.named_entities, (s_nan,)],
    ["noun_chunks", nlp.noun_chunks, (s_nan,)],
]

test_cases_preprocessing = [
    ["lowercase", preprocessing.lowercase, (s_nan,)],
    ["replace_digits", preprocessing.replace_digits, (s_nan, "")],
    ["remove_digits", preprocessing.remove_digits, (s_nan,)],
    ["replace_punctuation", preprocessing.replace_punctuation, (s_nan, "")],
    ["remove_punctuation", preprocessing.remove_punctuation, (s_nan,)],
    ["remove_diacritics", preprocessing.remove_diacritics, (s_nan,)],
    ["remove_whitespace", preprocessing.remove_whitespace, (s_nan,)],
    ["replace_stopwords", preprocessing.replace_stopwords, (s_nan, "")],
    ["remove_stopwords", preprocessing.remove_stopwords, (s_nan,)],
    ["stem", preprocessing.stem, (s_nan,)],
    ["remove_round_brackets", preprocessing.remove_round_brackets, (s_nan,)],
    ["remove_curly_brackets", preprocessing.remove_curly_brackets, (s_nan,)],
    ["remove_square_brackets", preprocessing.remove_square_brackets, (s_nan,)],
    ["remove_angle_brackets", preprocessing.remove_angle_brackets, (s_nan,)],
    ["remove_brackets", preprocessing.remove_brackets, (s_nan,)],
    ["remove_html_tags", preprocessing.remove_html_tags, (s_nan,)],
    ["tokenize", preprocessing.tokenize, (s_nan,)],
    ["tokenize_with_phrases", preprocessing.tokenize_with_phrases, (s_nan,)],
    ["replace_urls", preprocessing.replace_urls, (s_nan, "")],
    ["remove_urls", preprocessing.remove_urls, (s_nan,)],
    ["replace_tags", preprocessing.replace_tags, (s_nan, "")],
    ["remove_tags", preprocessing.remove_tags, (s_nan,)],
]

test_cases_representation = [
    [
        "term_frequency",
        representation.term_frequency,
        (preprocessing.tokenize(s_nan),),
    ],
    # ["tfidf", representation.tfidf, (preprocessing.tokenize(s_nan),)],
    ["pca", representation.pca, (s_numeric_and_nan_lists, 0)],
    ["nmf", representation.nmf, (s_numeric_and_nan_lists,)],
    ["tsne", representation.tsne, (s_numeric_and_nan_lists,)],
    ["kmeans", representation.kmeans, (s_numeric_and_nan_lists, 1)],
    ["dbscan", representation.dbscan, (s_numeric_and_nan_lists,)],
    ["meanshift", representation.meanshift, (s_numeric_and_nan_lists,)],
]

test_cases_visualization = []

test_cases = (
    test_cases_nlp
    + test_cases_preprocessing
    + test_cases_representation
    + test_cases_visualization
)


class AbstractNaNTest(PandasTestCase):
    """
    Class for np.NaN test cases. Tests for all cases
    in test_cases whether the function ignores an input
    with np.nan entries. Some function's tests
    are implemented manually as they take different inputs.

    """

    """
    Tests defined in test_cases above.
    """

    @parameterized.expand(test_cases)
    def test_ignores_nan(self, name, test_function, valid_input):
        s = valid_input[0]
        result_s = test_function(*valid_input)
        t_same = pd.Series(s.values)
        self.assertTrue(result_s.isna().equals(t_same.isna()))
