import pandas as pd
from texthero import nlp, visualization, preprocessing, representation
import texthero as hero

from . import PandasTestCase
import unittest
import string
from parameterized import parameterized
import importlib
import inspect
import numpy as np


"""
This file intends to test each function whether the input Series's index is the same as
the output Series.

This will go through all functions under texthero, and automatically 
generate test cases, according to the HeroSeries type they accept, e.g. TokenSeries, TextSeries, etc.

Normally, functions receives HeroSeries and returns Series is okay for auto-testing.
However there might be exceptions that you want to specify a test case manually, or omit a function
for testing. For example, 
- Functions that needs multiple arguments (preprocessing.replace_stopwords)
- Functions that returns Series with different index (representation.tfidf)
- Functions that doesn't give Series as output (mostly in visualization)
- Functions that doesn't take HeroSeries (yet)

In those cases, you can add your custom test case so as to override the default one, 
in the form of [name_of_test_case, function_to_test, tuple_of_valid_input_for_the_function]. 
If you want to omit some functions, add their string name to func_white_list variable.

The tests will be run by AbstractIndexTest below through the @parameterized
decorator. The names will be expanded automatically, so e.g. "named_entities"
creates test cases test_correct_index_named_entities and test_incorrect_index_named_entities.
"""

# Define the valid input for each HeroSeries type
s_text = pd.Series(["Test"], index=[5])
s_tokenized_lists = pd.Series([["Test", "Test2"], ["Test3"]], index=[5, 6])
s_numeric = pd.Series([5.0], index=[5])
s_numeric_lists = pd.Series([[5.0, 5.0], [6.0, 6.0]], index=[5, 6])

valid_inputs = {
    "TokenSeries": s_tokenized_lists,
    "TextSeries": s_text,
    "VectorSeries": s_numeric_lists,
}

# Specify your custom test cases here (functions that
# has multiple arguments, doesn't accpet HeroSeries, etc.)
test_cases_nlp = []

test_cases_preprocessing = [
    ["replace_digits", preprocessing.replace_digits, (s_text, "")],
    ["replace_punctuation", preprocessing.replace_punctuation, (s_text, "")],
    ["replace_stopwords", preprocessing.replace_stopwords, (s_text, "")],
    ["replace_urls", preprocessing.replace_urls, (s_text, "")],
    ["replace_tags", preprocessing.replace_tags, (s_text, "")],
    ["replace_hashtags", preprocessing.replace_hashtags, (s_text, "")],
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

# Custom test cases, a dictionary of {func_str: test_case}
test_case_custom = {}
for case in (
    test_cases_nlp
    + test_cases_preprocessing
    + test_cases_representation
    + test_cases_visualization
):
    test_case_custom[case[0]] = case


# Put functions' name into white list if you want to omit them
# func_white_list = {
# 'scatterplot',
# 'wordcloud',
# 'top_words'
# }
func_white_list = set(
    [s for s in inspect.getmembers(visualization, inspect.isfunction)]
)

test_cases = []

# Find all functions under texthero
func_strs = [
    s[0]
    for s in inspect.getmembers(hero, inspect.isfunction)
    if s not in func_white_list
]

for func_str in func_strs:
    # Use a custom test case
    if func_str in test_case_custom:
        test_cases.append(test_case_custom[func_str])
    else:
        # Generate one by default
        func = getattr(hero, func_str)
        # Functions accept HeroSeries
        if (
            hasattr(func, "allowed_hero_series_type")
            and func.allowed_hero_series_type.__name__ in valid_inputs
        ):
            test_cases.append(
                [
                    func_str,
                    func,
                    (valid_inputs[func.allowed_hero_series_type.__name__],),
                ]
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
