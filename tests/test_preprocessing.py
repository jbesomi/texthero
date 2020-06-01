import string

import pandas as pd
import numpy as np
import doctest

from texthero import preprocessing, stopwords
from . import PandasTestCase


"""
Test doctest
"""


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(preprocessing))
    return tests


class TestPreprocessing(PandasTestCase):
    """
    Test remove digits.
    """

    def test_remove_digits_only_block(self):
        s = pd.Series("remove block of digits 1234 h1n1")
        s_true = pd.Series("remove block of digits   h1n1")
        self.assertEqual(preprocessing.remove_digits(s), s_true)

    def test_remove_digits_any(self):
        s = pd.Series("remove block of digits 1234 h1n1")
        s_true = pd.Series("remove block of digits   h n ")
        self.assertEqual(preprocessing.remove_digits(s, only_blocks=False), s_true)

    def test_remove_digits_brackets(self):
        s = pd.Series("Digits in bracket (123 $) needs to be cleaned out")
        s_true = pd.Series("Digits in bracket (  $) needs to be cleaned out")
        self.assertEqual(preprocessing.remove_digits(s), s_true)

    def test_remove_digits_start(self):
        s = pd.Series("123 starting digits needs to be cleaned out")
        s_true = pd.Series("  starting digits needs to be cleaned out")
        self.assertEqual(preprocessing.remove_digits(s), s_true)

    def test_remove_digits_end(self):
        s = pd.Series("end digits needs to be cleaned out 123")
        s_true = pd.Series("end digits needs to be cleaned out  ")
        self.assertEqual(preprocessing.remove_digits(s), s_true)

    def test_remove_digits_phone(self):
        s = pd.Series("+41 1234 5678")
        s_true = pd.Series("+     ")
        self.assertEqual(preprocessing.remove_digits(s), s_true)

    def test_remove_digits_punctuation(self):
        s = pd.Series(string.punctuation)
        s_true = pd.Series(string.punctuation)
        self.assertEqual(preprocessing.remove_digits(s), s_true)

    """
    Test replace digits
    """

    def test_replace_digits(self):
        s = pd.Series("1234 falcon9")
        s_true = pd.Series("X falcon9")
        self.assertEqual(preprocessing.replace_digits(s, "X"), s_true)

    def test_replace_digits_any(self):
        s = pd.Series("1234 falcon9")
        s_true = pd.Series("X falconX")
        self.assertEqual(
            preprocessing.replace_digits(s, "X", only_blocks=False), s_true
        )

    """
    Remove punctuation.
    """

    def test_remove_punctation(self):
        s = pd.Series("Remove all! punctuation!! ()")
        s_true = pd.Series(
            "Remove all  punctuation   "
        )  # TODO maybe just remove space?
        self.assertEqual(preprocessing.remove_punctuation(s), s_true)

    """
    Remove diacritics.
    """

    def test_remove_diactitics(self):
        s = pd.Series("hèllo")
        s_true = pd.Series("hello")
        self.assertEqual(preprocessing.remove_diacritics(s), s_true)

    """
    Remove whitespace.
    """

    def test_remove_whitespace(self):
        s = pd.Series("hello   world  hello        world ")
        s_true = pd.Series("hello world hello world")
        self.assertEqual(preprocessing.remove_whitespace(s), s_true)

    """
    Test pipeline.
    """

    def test_pipeline_stopwords(self):
        s = pd.Series("E-I-E-I-O\nAnd on")
        s_true = pd.Series("e-i-e-i-o\n ")
        pipeline = [preprocessing.lowercase, preprocessing.remove_stopwords]
        self.assertEqual(preprocessing.clean(s, pipeline=pipeline), s_true)

    """
    Test stopwords.
    """

    def test_remove_stopwords(self):
        text = "i am quite intrigued"
        text_default_preprocessed = "  quite intrigued"
        text_spacy_preprocessed = "   intrigued"
        text_custom_preprocessed = "i  quite "

        self.assertEqual(
            preprocessing.remove_stopwords(pd.Series(text)),
            pd.Series(text_default_preprocessed),
        )
        self.assertEqual(
            preprocessing.remove_stopwords(
                pd.Series(text), stopwords=stopwords.SPACY_EN
            ),
            pd.Series(text_spacy_preprocessed),
        )
        self.assertEqual(
            preprocessing.remove_stopwords(
                pd.Series(text), stopwords={"am", "intrigued"}
            ),
            pd.Series(text_custom_preprocessed),
        )

    def test_stopwords_are_set(self):
        self.assertEqual(type(stopwords.DEFAULT), set)
        self.assertEqual(type(stopwords.NLTK_EN), set)
        self.assertEqual(type(stopwords.SPACY_EN), set)

    """
    Test remove html tags
    """

    def test_remove_html_tags(self):
        s = pd.Series("<html>remove <br>html</br> tags<html> &nbsp;")
        s_true = pd.Series("remove html tags ")
        self.assertEqual(preprocessing.remove_html_tags(s), s_true)

    """
    Text tokenization
    """

    def test_tokenize(self):
        s = pd.Series("text to tokenize")
        s_true = pd.Series([["text", "to", "tokenize"]])
        self.assertEqual(preprocessing.tokenize(s), s_true)

    def test_tokenize_multirows(self):
        s = pd.Series(["first row", "second row"])
        s_true = pd.Series([["first", "row"], ["second", "row"]])
        self.assertEqual(preprocessing.tokenize(s), s_true)

    def test_tokenize_split_punctuation(self):
        s = pd.Series(["ready. set, go!"])
        s_true = pd.Series([["ready", ".", "set", ",", "go", "!"]])
        self.assertEqual(preprocessing.tokenize(s), s_true)

    def test_tokenize_not_split_in_between_punctuation(self):
        s = pd.Series(["don't say hello-world"])
        s_true = pd.Series([["don't", "say", "hello-world"]])
        self.assertEqual(preprocessing.tokenize(s), s_true)

    """
     Has content
    """

    def test_has_content(self):
        s = pd.Series(["c", np.nan, "\t\n", " ", "", "has content", None])
        s_true = pd.Series([True, False, False, False, False, True, False])
        self.assertEqual(preprocessing.has_content(s), s_true)

    """
    Test remove urls
    """

    def test_remove_urls(self):
        s = pd.Series("http://tests.com http://www.tests.com")
        s_true = pd.Series("   ")
        self.assertEqual(preprocessing.remove_urls(s), s_true)

    def test_remove_urls_https(self):
        s = pd.Series("https://tests.com https://www.tests.com")
        s_true = pd.Series("   ")
        self.assertEqual(preprocessing.remove_urls(s), s_true)

    def test_remove_urls_multiline(self):
        s = pd.Series("https://tests.com \n https://tests.com")
        s_true = pd.Series("  \n  ")
        self.assertEqual(preprocessing.remove_urls(s), s_true)

    """
    Remove brackets
    """

    def test_remove_round_brackets(self):
        s = pd.Series("Remove all (brackets)(){/}[]<>")
        s_true = pd.Series("Remove all {/}[]<>")
        self.assertEqual(preprocessing.remove_round_brackets(s), s_true)

    def test_remove_curly_brackets(self):
        s = pd.Series("Remove all (brackets)(){/}[]<> { }")
        s_true = pd.Series("Remove all (brackets)()[]<> ")
        self.assertEqual(preprocessing.remove_curly_brackets(s), s_true)

    def test_remove_square_brackets(self):
        s = pd.Series("Remove all [brackets](){/}[]<>")
        s_true = pd.Series("Remove all (){/}<>")
        self.assertEqual(preprocessing.remove_square_brackets(s), s_true)

    def test_remove_angle_brackets(self):
        s = pd.Series("Remove all <brackets>(){/}[]<>")
        s_true = pd.Series("Remove all (){/}[]")
        self.assertEqual(preprocessing.remove_angle_brackets(s), s_true)

    def test_remove_brackets(self):
        s = pd.Series(
            "Remove all [square_brackets]{/curly_brackets}(round_brackets)<angle_brackets>"
        )
        s_true = pd.Series("Remove all ")
        self.assertEqual(preprocessing.remove_brackets(s), s_true)
