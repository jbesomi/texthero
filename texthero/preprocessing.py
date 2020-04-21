"""
Text preprocessing
"""

import re
import string
import unidecode
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

def fillna_s(input: pd.Series) -> pd.Series:
    """
    Fillna the given `input` of the pd.Series with ""
    """
    input.fillna("")
    return input.astype("str")


def lowercase_s(input: pd.Series) -> pd.Series:
    """
    Lowercase the given `input` of the pd.Series
    """
    return input.str.lower()

def remove_digits_s(input: pd.Series, only_blocks=True) -> pd.Series:
    """
    Remove digits from series

    Parameters
    ----------

    input : pd.Series
    only_blocks : bool
        Remove only blocks of digits. For instance, `hel1234lo 1234` becomes `hel1234lo`.

    Returns
    -------

    pd.Series

    Examples
    --------

        >>> import texthero
        >>> s = pd.Series(["remove_digits_s remove all the 1234 digits of a pandas series. H1N1"])
        >>> texthero.preprocessing.remove_digits_s(s)
        u'remove_digits_s remove all the digits of a pandas series. H1N1'
        >>> texthero.preprocessing.remove_digits_s(s, only_blocks=False)
        u'remove_digits_s remove all the digits of a pandas series. HN'
    """

    if type(input) is not pd.Series:
        raise ValueError('input arguments must be of type pd.Series')

    if only_blocks:
        return input.str.replace(r"\s+\d+\s+", " ")
    else:
        return input.str.replace(r"\d+", " ")


def remove_punctuations_s(input : pd.Series) -> pd.Series:
    """
    Remove punctuations from input
    """
    RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
    return input.str.replace(RE_PUNCT, " ")

def remove_diacritics_s(input: pd.Series) -> pd.Series:
    """
    Remove diacritics (as accent marks) from input
    """
    return input.apply(unidecode.unidecode)

def remove_spaces_s(input: pd.Series) -> pd.Series:
    """
    Remove any type of space between words.
    """

    return input.str.replace(u"\xa0", u" ").str.split().str.join(" ")

def remove_stop_words_s(input: pd.Series) -> pd.Series:

    stop_words = set(stopwords.words("english"))
    pat = r'\b(?:{})\b'.format('|'.join(stop_words))
    return input.str.replace(pat, '')


def stemm(text):
    """Stem words"""
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text])

def stemm_s(input: pd.Series) -> pd.Series:
    return input.str.split().apply(stemm)

def get_default_pipeline():
    """
    Default pipeline:
        - remove_lowercase
        - remove_numbers
        - remove_punctuations
        - remove_diacritics
        - remove_white_space
        - remove_stop_words
        - stemming
    """
    return [fillna_s,
            lowercase_s,
            remove_digits_s,
            remove_punctuations_s,
            remove_diacritics_s,
            remove_spaces_s,
            remove_stop_words_s,
            ]

def apply_fun_to_obj(fun, obj, text_columns):
    for col in text_columns:
        obj[col] = fun(obj[col])
    return obj

def do_preprocess(df, text_columns=['text'], pipeline=None):

    if not pipeline:
        pipeline = get_default_pipeline()

    def clean(text):
        if text is None:
            return text
        for f in pipeline:
            text = f(text)
        return text

    if isinstance(text_columns, str):
        df[text_columns + "_clean"] = clean(df[text_columns])
    else:
        for col in text_columns:
            df[col + "_clean"] = clean(df[col])

    return df
