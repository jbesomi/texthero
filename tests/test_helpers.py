"""
Unit-tests for the helper module.
"""

import pandas as pd
import numpy as np

from . import PandasTestCase
import doctest
import unittest
import warnings
import string

from texthero import helper, preprocessing, nlp

"""
Doctests.
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(helper))
    return tests


"""
Test Decorators.
"""


class TestHelpers(PandasTestCase):
    """
    handle_nans.
    """

    def test_handle_nans(self):
        s = pd.Series(["Test", np.nan, pd.NA])

        @helper.handle_nans(replace_nans_with="This was a NAN")
        def f(s):
            return s

        s_true = pd.Series(["Test", "This was a NAN", "This was a NAN"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(f(s), s_true)

        with self.assertWarns(Warning):
            f(s)

    def test_handle_nans_no_nans_in_input(self):
        s = pd.Series(["Test"])

        @helper.handle_nans(replace_nans_with="This was a NAN")
        def f(s):
            return s

        s_true = pd.Series(["Test"])

        self.assertEqual(f(s), s_true)

    # This is not in test_indexes.py as it requires a custom test case.
    def test_handle_nans_index(self):
        s = pd.Series(["Test", np.nan, pd.NA], index=[4, 5, 6])

        @helper.handle_nans(replace_nans_with="This was a NAN")
        def f(s):
            return s

        s_true = pd.Series(
            ["Test", "This was a NAN", "This was a NAN"], index=[4, 5, 6]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertTrue(f(s).index.equals(s_true.index))


class TestPreprocessingParallelized(PandasTestCase):
    """
    Test remove digits.
    """

    def setUp(self):
        helper.MIN_LINES_FOR_PARALLELIZATION = 0
        helper.PARALLELIZE = True

    def tearDown(self):
        helper.MIN_LINES_FOR_PARALLELIZATION = 10000
        helper.PARALLELIZE = True

    def parallelized_test_helper(self, func, s, non_parallel_s_true, **kwargs):

        s = s
        non_parallel_s_true = non_parallel_s_true

        pd.testing.assert_series_equal(non_parallel_s_true, func(s, **kwargs))

    def test_remove_digits_only_block(self):
        s = pd.Series("remove block of digits 1234 h1n1")
        s_true = pd.Series("remove block of digits   h1n1")
        self.parallelized_test_helper(preprocessing.remove_digits, s, s_true)

    def test_remove_digits_any(self):
        s = pd.Series("remove block of digits 1234 h1n1")
        s_true = pd.Series("remove block of digits   h n ")

        self.parallelized_test_helper(
            preprocessing.remove_digits, s, s_true, only_blocks=False
        )

    def test_remove_digits_brackets(self):
        s = pd.Series("Digits in bracket (123 $) needs to be cleaned out")
        s_true = pd.Series("Digits in bracket (  $) needs to be cleaned out")
        self.parallelized_test_helper(preprocessing.remove_digits, s, s_true)

    def test_remove_digits_start(self):
        s = pd.Series("123 starting digits needs to be cleaned out")
        s_true = pd.Series("  starting digits needs to be cleaned out")
        self.parallelized_test_helper(preprocessing.remove_digits, s, s_true)

    def test_remove_digits_end(self):
        s = pd.Series("end digits needs to be cleaned out 123")
        s_true = pd.Series("end digits needs to be cleaned out  ")
        self.parallelized_test_helper(preprocessing.remove_digits, s, s_true)

    def test_remove_digits_phone(self):
        s = pd.Series("+41 1234 5678")
        s_true = pd.Series("+     ")
        self.parallelized_test_helper(preprocessing.remove_digits, s, s_true)

    def test_remove_digits_punctuation(self):
        s = pd.Series(string.punctuation)
        s_true = pd.Series(string.punctuation)
        self.parallelized_test_helper(preprocessing.remove_digits, s, s_true)

    """
    Test replace digits
    """

    def test_replace_digits(self):
        s = pd.Series("1234 falcon9")
        s_true = pd.Series("X falcon9")
        self.parallelized_test_helper(
            preprocessing.replace_digits, s, s_true, symbols="X"
        )

    def test_replace_digits_any(self):
        s = pd.Series("1234 falcon9")
        s_true = pd.Series("X falconX")
        self.parallelized_test_helper(
            preprocessing.replace_digits, s, s_true, symbols="X", only_blocks=False
        )

    """
    Remove punctuation.
    """

    def test_remove_punctation(self):
        s = pd.Series("Remove all! punctuation!! ()")
        s_true = pd.Series(
            "Remove all  punctuation   "
        )  # TODO maybe just remove space?
        self.parallelized_test_helper(preprocessing.remove_punctuation, s, s_true)

    """
    Remove diacritics.
    """

    def test_remove_diactitics(self):
        s = pd.Series("Montréal, über, 12.89, Mère, Françoise, noël, 889, اِس, اُس")
        s_true = pd.Series("Montreal, uber, 12.89, Mere, Francoise, noel, 889, اس, اس")
        self.parallelized_test_helper(preprocessing.remove_diacritics, s, s_true)

    """
    Remove whitespace.
    """

    def test_remove_whitespace(self):
        s = pd.Series("hello   world  hello        world ")
        s_true = pd.Series("hello world hello world")
        self.parallelized_test_helper(preprocessing.remove_whitespace, s, s_true)

    """
    Test pipeline.
    """

    def test_pipeline_stopwords(self):
        s = pd.Series("E-I-E-I-O\nAnd on")
        s_true = pd.Series("e-i-e-i-o\n ")
        pipeline = [preprocessing.lowercase, preprocessing.remove_stopwords]
        self.parallelized_test_helper(preprocessing.clean, s, s_true, pipeline=pipeline)

    """
    Test remove html tags
    """

    def test_remove_html_tags(self):
        s = pd.Series("<html>remove <br>html</br> tags<html> &nbsp;")
        s_true = pd.Series("remove html tags ")
        self.parallelized_test_helper(preprocessing.remove_html_tags, s, s_true)

    """
    Text tokenization
    """

    def test_tokenize(self):
        s = pd.Series("text to tokenize")
        s_true = pd.Series([["text", "to", "tokenize"]])
        self.parallelized_test_helper(preprocessing.tokenize, s, s_true)

    """
    Has content
    """

    def test_has_content(self):
        s = pd.Series(["c", np.nan, "\t\n", " ", "", "has content", None])
        s_true = pd.Series([True, False, False, False, False, True, False])
        self.parallelized_test_helper(preprocessing.has_content, s, s_true)

    """
    Test remove urls
    """

    def test_remove_urls(self):
        s = pd.Series("http://tests.com http://www.tests.com")
        s_true = pd.Series("   ")
        self.parallelized_test_helper(preprocessing.remove_urls, s, s_true)

    """
    Remove brackets
    """

    def test_remove_brackets(self):
        s = pd.Series(
            "Remove all [square_brackets]{/curly_brackets}(round_brackets)<angle_brackets>"
        )
        s_true = pd.Series("Remove all ")
        self.parallelized_test_helper(preprocessing.remove_brackets, s, s_true)

    """
    Test replace and remove tags
    """

    def test_replace_tags(self):
        s = pd.Series("Hi @tag, we will replace you")
        s_true = pd.Series("Hi TAG, we will replace you")
        self.parallelized_test_helper(
            preprocessing.replace_tags, s, s_true, symbol="TAG"
        )

    def test_remove_tags_alphabets(self):
        s = pd.Series("Hi @tag, we will remove you")
        s_true = pd.Series("Hi  , we will remove you")

        self.parallelized_test_helper(preprocessing.remove_tags, s, s_true)

    """
    Test replace and remove hashtags
    """

    def test_replace_hashtags(self):
        s = pd.Series("Hi #hashtag, we will replace you")
        s_true = pd.Series("Hi HASHTAG, we will replace you")

        self.parallelized_test_helper(
            preprocessing.replace_hashtags, s, s_true, symbol="HASHTAG"
        )

    def test_remove_hashtags(self):
        s = pd.Series("Hi #hashtag_trending123, we will remove you")
        s_true = pd.Series("Hi  , we will remove you")

        self.parallelized_test_helper(preprocessing.remove_hashtags, s, s_true)

    """
    Test NLP for parallelization
    """

    """
    Named entity.
    """

    def test_named_entities(self):
        s = pd.Series("New York is a big city")
        s_true = pd.Series([[("New York", "GPE", 0, 8)]])
        self.parallelized_test_helper(nlp.named_entities, s, s_true)

    """
    Noun chunks.
    """

    def test_noun_chunks(self):
        s = pd.Series("Today is such a beautiful day")
        s_true = pd.Series(
            [[("Today", "NP", 0, 5), ("such a beautiful day", "NP", 9, 29)]]
        )

        self.parallelized_test_helper(nlp.noun_chunks, s, s_true)

    """
    Count sentences.
    """

    def test_count_sentences(self):
        s = pd.Series("I think ... it counts correctly. Doesn't it? Great!")
        s_true = pd.Series(3)
        self.parallelized_test_helper(nlp.count_sentences, s, s_true)

    """
    POS tagging.
    """

    def test_pos(self):
        s = pd.Series(["Today is such a beautiful day", "São Paulo is a great city"])

        s_true = pd.Series(
            [
                [
                    ("Today", "NOUN", "NN", 0, 5),
                    ("is", "AUX", "VBZ", 6, 8),
                    ("such", "DET", "PDT", 9, 13),
                    ("a", "DET", "DT", 14, 15),
                    ("beautiful", "ADJ", "JJ", 16, 25),
                    ("day", "NOUN", "NN", 26, 29),
                ],
                [
                    ("São", "PROPN", "NNP", 0, 3),
                    ("Paulo", "PROPN", "NNP", 4, 9),
                    ("is", "AUX", "VBZ", 10, 12),
                    ("a", "DET", "DT", 13, 14),
                    ("great", "ADJ", "JJ", 15, 20),
                    ("city", "NOUN", "NN", 21, 25),
                ],
            ]
        )

        self.parallelized_test_helper(nlp.pos_tag, s, s_true)
