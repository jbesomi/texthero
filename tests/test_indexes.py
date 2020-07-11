import pandas as pd
from texthero import nlp, visualization, preprocessing, representation

from . import PandasTestCase
import unittest
import string
from parameterized import parameterized


# Define valid inputs for different functions.
s_text = pd.Series(["Test"], index=[5])
s_numeric = pd.Series([5.0], index=[5])
s_numeric_lists = pd.Series([[5.0, 5.0]], index=[5])

# Define all test cases. Every test case is a list
# of [name of test case, function to test, valid input for the function].
# The tests will be run by AbstractIndexTest below through the @parameterized
# decorator.
# The names will be expanded automatically, so e.g. "named_entities"
# creates test cases test_correct_index_named_entities and test_incorrect_index_named_entities.

test_cases_nlp = [
    ["named_entities", nlp.named_entities, s_text],
    ["noun_chunks", nlp.noun_chunks, s_text],
]

test_cases_preprocessing = [
    ["fillna", preprocessing.fillna, s_text],
    ["lowercase", preprocessing.lowercase, s_text],
    ["replace_digits", preprocessing.replace_digits, s_text],
    ["remove_digits", preprocessing.remove_digits, s_text],
    ["replace_punctuation", preprocessing.replace_punctuation, s_text],
    ["remove_punctuation", preprocessing.remove_punctuation, s_text],
    ["remove_diacritics", preprocessing.remove_diacritics, s_text],
    ["remove_whitespace", preprocessing.remove_whitespace, s_text],
    ["replace_stopwords", preprocessing.replace_stopwords, s_text],
    ["remove_stopwords", preprocessing.remove_stopwords, s_text],
    ["stem", preprocessing.stem, s_text],
    ["clean", preprocessing.clean, s_text],
    ["remove_round_brackets", preprocessing.remove_round_brackets, s_text],
    ["remove_curly_brackets", preprocessing.remove_curly_brackets, s_text],
    ["remove_square_brackets", preprocessing.remove_square_brackets, s_text],
    ["remove_angle_brackets", preprocessing.remove_angle_brackets, s_text],
    ["remove_brackets", preprocessing.remove_brackets, s_text],
    ["remove_html_tags", preprocessing.remove_html_tags, s_text],
    ["tokenize", preprocessing.tokenize, s_text],
    ["tokenize_with_phrases", preprocessing.tokenize_with_phrases, s_text],
    ["replace_urls", preprocessing.replace_urls, s_text],
    ["remove_urls", preprocessing.remove_urls, s_text],
    ["replace_tags", preprocessing.replace_tags, s_text],
    ["remove_tags", preprocessing.remove_tags, s_text],
]

test_cases_representation = [
    ["term_frequency", representation.term_frequency, s_text],
    ["tfidf", representation.tfidf, s_text],
    ["pca", representation.pca, s_text],
    ["nmf", representation.nmf, s_text],
    ["tsne", representation.tsne, s_text],
    ["kmeans", representation.kmeans, s_text],
    ["dbscan", representation.dbscan, s_text],
    ["meanshift", representation.meanshift, s_text],
]

test_cases_visualization = [["top_words", visualization.top_words, s_text]]


class AbstractIndexTest(PandasTestCase):
    """
    Class for index test cases. Tests for all cases
    in test_cases whether the input's index is correctly
    preserved by the function. Some function's tests
    are implemented manually as they take different inputs.

    """

    @parameterized.expand(test_cases_nlp)
    def test_correct_index(self, name, test_function, valid_input):
        s = valid_input
        result_s = test_function(s)
        t_same_index = pd.Series(s.values, s.index)
        self.assertTrue(result_s.index.equals(t_same_index.index))

    @parameterized.expand(test_cases_nlp)
    def test_incorrect_index(self, name, test_function, valid_input):
        s = valid_input
        result_s = test_function(s)
        t_different_index = pd.Series(s.values, index=None)
        self.assertFalse(result_s.index.equals(t_different_index.index))

    def test_correct_index_most_similar(self):
        s = pd.DataFrame([[1.0], [2.0]], index=["word1", "word2"])
        result_s = representation.most_similar(s, "word1")
        t_same_index = pd.DataFrame(s.values, s.index)
        self.assertTrue(result_s.index.equals(t_same_index.index))

    def test_incorrect_index_most_similar(self):
        s = pd.DataFrame([[1.0], [2.0]], index=["word1", "word2"])
        result_s = representation.most_similar(s, "word1")
        t_different_index = pd.DataFrame(s.values, index=None)
        self.assertFalse(result_s.index.equals(t_different_index.index))
