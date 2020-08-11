import pandas as pd
from texthero import nlp, visualization, preprocessing, representation

from . import PandasTestCase
import unittest
import string
from parameterized import parameterized


# Define valid inputs for different functions.
s_text = pd.Series(["Test"], index=[5])
s_tokenized_lists = pd.Series([["Test", "Test2"], ["Test3"]], index=[5, 6])
s_numeric = pd.Series([5.0], index=[5])
s_numeric_lists = pd.Series([[5.0, 5.0], [6.0, 6.0]], index=[5, 6])

# Define all test cases. Every test case is a list
# of [name of test case, function to test, tuple of valid input for the function].
# First argument of valid input has to be the Pandas Series where we
# want to keep the index. If this is different for a function, a separate
# test case has to implemented in the class below.
# The tests will be run by AbstractIndexTest below through the @parameterized
# decorator.
# The names will be expanded automatically, so e.g. "named_entities"
# creates test cases test_correct_index_named_entities and test_incorrect_index_named_entities.

test_cases_nlp = [
    ["named_entities", nlp.named_entities, (s_text,)],
    ["noun_chunks", nlp.noun_chunks, (s_text,)],
]

test_cases_preprocessing = [
    ["fillna", preprocessing.fillna, (s_text,)],
    ["lowercase", preprocessing.lowercase, (s_text,)],
    ["replace_digits", preprocessing.replace_digits, (s_text, "")],
    ["remove_digits", preprocessing.remove_digits, (s_text,)],
    ["replace_punctuation", preprocessing.replace_punctuation, (s_text, "")],
    ["remove_punctuation", preprocessing.remove_punctuation, (s_text,)],
    ["remove_diacritics", preprocessing.remove_diacritics, (s_text,)],
    ["remove_whitespace", preprocessing.remove_whitespace, (s_text,)],
    ["replace_stopwords", preprocessing.replace_stopwords, (s_text, "")],
    ["remove_stopwords", preprocessing.remove_stopwords, (s_text,)],
    ["stem", preprocessing.stem, (s_text,)],
    ["clean", preprocessing.clean, (s_text,)],
    ["remove_round_brackets", preprocessing.remove_round_brackets, (s_text,)],
    ["remove_curly_brackets", preprocessing.remove_curly_brackets, (s_text,)],
    ["remove_square_brackets", preprocessing.remove_square_brackets, (s_text,)],
    ["remove_angle_brackets", preprocessing.remove_angle_brackets, (s_text,)],
    ["remove_brackets", preprocessing.remove_brackets, (s_text,)],
    ["remove_html_tags", preprocessing.remove_html_tags, (s_text,)],
    ["tokenize", preprocessing.tokenize, (s_text,)],
    ["phrases", preprocessing.phrases, (s_tokenized_lists,)],
    ["replace_urls", preprocessing.replace_urls, (s_text, "")],
    ["remove_urls", preprocessing.remove_urls, (s_text,)],
    ["replace_tags", preprocessing.replace_tags, (s_text, "")],
    ["remove_tags", preprocessing.remove_tags, (s_text,)],
]

test_cases_representation = [
    [
        "count",
        lambda x: representation.flatten(representation.count(x)),
        (s_tokenized_lists,),
    ],
    [
        "term_frequency",
        lambda x: representation.flatten(representation.term_frequency(x)),
        (s_tokenized_lists,),
    ],
    [
        "tfidf",
        lambda x: representation.flatten(representation.tfidf(x)),
        (s_tokenized_lists,),
    ],
    ["pca", representation.pca, (s_numeric_lists, 0)],
    ["nmf", representation.nmf, (s_numeric_lists,)],
    ["tsne", representation.tsne, (s_numeric_lists,)],
    ["kmeans", representation.kmeans, (s_numeric_lists, 1)],
    ["dbscan", representation.dbscan, (s_numeric_lists,)],
    ["meanshift", representation.meanshift, (s_numeric_lists,)],
]

test_cases_visualization = []

test_cases = (
    test_cases_nlp
    + test_cases_preprocessing
    + test_cases_representation
    + test_cases_visualization
)


class AbstractIndexTest(PandasTestCase):
    """
    Class for index test cases. Tests for all cases
    in test_cases whether the input's index is correctly
    preserved by the function. Some function's tests
    are implemented manually as they take different inputs.

    """

    """
    Tests defined in test_cases above.
    """

    @parameterized.expand(test_cases)
    def test_correct_index(self, name, test_function, valid_input):
        s = valid_input[0]
        result_s = test_function(*valid_input)
        t_same_index = pd.Series(s.values, s.index)
        self.assertTrue(result_s.index.equals(t_same_index.index))

    @parameterized.expand(test_cases)
    def test_incorrect_index(self, name, test_function, valid_input):
        s = valid_input[0]
        result_s = test_function(*valid_input)
        t_different_index = pd.Series(s.values, index=None)
        self.assertFalse(result_s.index.equals(t_different_index.index))
