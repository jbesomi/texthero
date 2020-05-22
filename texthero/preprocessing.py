"""
Preprocess text-based Pandas DataFrame.
"""
import re
import string
from typing import Optional, Set

import numpy as np
import pandas as pd
import unidecode
from nltk.stem import PorterStemmer, SnowballStemmer

from texthero import stopwords as _stopwords


def fillna(input: pd.Series) -> pd.Series:
    """Replace not assigned values with empty spaces."""
    return input.fillna("").astype("str")


def lowercase(input: pd.Series) -> pd.Series:
    """Lowercase all text."""
    return input.str.lower()


def _remove_block_digits(text):
    """
    Remove block of digits from text.

    Examples
    --------
    >>> _remove_block_digits("hi 123")
    'hi '
    """
    pattern = r"""(?x)                          # set flag to allow verbose regexps
      \w+(?:-\w+)*                              # words with optional internal hyphens
      | \$?\d+(?:\.\d+)?%?                      # currency and percentages, e.g. $12.40, 82%
      | [][!"#$%&'*+,-./:;<=>?@\\^():_`{|}~]    # these are separate tokens; includes ], [
      | \s*
    """
    return "".join(t for t in re.findall(pattern, text) if not t.isnumeric())


def remove_digits(input: pd.Series, only_blocks=True) -> pd.Series:
    """
    Remove all digits from a series and replace it with an empty space.

    Parameters
    ----------
    input : pd.Series
    only_blocks : bool
                  Remove only blocks of digits. For instance, `hel1234lo 1234` becomes `hel1234lo`.

    # Examples
    ----------
    >>> s = pd.Series("7ex7hero is fun 1111")
    >>> remove_digits(s)
    0    7ex7hero is fun 
    dtype: object
    >>> remove_digits(s, only_blocks=False)
    0    exhero is fun 
    dtype: object
    """

    if only_blocks:
        return input.apply(_remove_block_digits)
    else:
        return input.str.replace(r"\d+", "")


def remove_punctuation(input: pd.Series) -> pd.Series:
    """
    Remove string.punctuation (!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).

    Replace it with a single space.
    """
    return replace_punctuation(input, " ")


def replace_punctuation(input: pd.Series, symbol: str) -> pd.Series:
    """
    Replace string.punctuation (!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~) with symbol argument.
    """

    RE_PUNCT = re.compile(r"([%s])+" % re.escape(string.punctuation), re.UNICODE)
    return input.str.replace(RE_PUNCT, symbol)


def remove_diacritics(input: pd.Series) -> pd.Series:
    """
    Remove all diacritics.
    """
    return input.apply(unidecode.unidecode)


def remove_whitespace(input: pd.Series) -> pd.Series:
    """
    Remove all extra white spaces between words.
    """

    return input.str.replace(u"\xa0", u" ").str.split().str.join(" ")


def _remove_stopwords(text: str, words: Set[str]) -> str:
    """
    Remove block of digits from text.

    Examples
    --------
    """

    pattern = r"""(?x)                          # Set flag to allow verbose regexps
      \w+(?:-\w+)*                              # Words with optional internal hyphens 
      | \s*                                     # Any space
      | [][!"#$%&'*+,-./:;<=>?@\\^():_`{|}~]    # Any symbol 
    """

    return "".join(t for t in re.findall(pattern, text) if t not in words)


def replace_stopwords(
    input: pd.Series, symbol: str, stopwords: Optional[Set[str]] = None
) -> pd.Series:
    """
    Replace all stopwords with symbol.

    By default uses NLTK's english stopwords of 179 words.

    Examples
    --------
    >>> s = pd.Series("the book of the jungle")
    >>> replace_stopwords(s, "X")
    0     book   jungle
    dtype: object

    """

    # FIX ME. Replace with custom symbol is not working and the docstring example is clearly wrong.

    if stopwords is None:
        stopwords = _stopwords.DEFAULT
    return input.apply(_remove_stopwords, args=(stopwords,))


def remove_stopwords(
    input: pd.Series, stopwords: Optional[Set[str]] = None
) -> pd.Series:
    """
    Remove all instances of `words` and replace it with an empty space.

    By default uses NLTK's english stopwords of 179 words.
    """
    return replace_stopwords(input, symbol=" ", stopwords=stopwords)


def stem(input: pd.Series, stem="snowball", language="english") -> pd.Series:
    """
    Stem series using either 'porter' or 'snowball' NLTK stemmers.

    Not in the default pipeline.

    Parameters
    ----------
    input
    stem
        Can be either 'snowball' or 'porter'. ("snowball" is default)
    language
        Supported languages: 
            danish dutch english finnish french german hungarian italian 
            norwegian portuguese romanian russian spanish swedish
    """

    if stem is "porter":
        stemmer = PorterStemmer()
    elif stem is "snowball":
        stemmer = SnowballStemmer(language)
    else:
        raise ValueError("stem argument must be either 'porter' of 'stemmer'")

    def _stem(text):
        """Stem words"""
        return " ".join([stemmer.stem(word) for word in text])

    return input.str.split().apply(_stem)


def get_default_pipeline() -> []:
    """
    Return a list contaning all the methods used in the default cleaning pipeline.

    Return a list with the following functions:
    - fillna
    - lowercase
    - remove_digits
    - remove_punctuation
    - remove_diacritics
    - remove_stopwords
    - remove_whitespace
    """
    return [
        fillna,
        lowercase,
        remove_digits,
        remove_punctuation,
        remove_diacritics,
        remove_stopwords,
        remove_whitespace,
    ]


def clean(s: pd.Series, pipeline=None) -> pd.Series:
    """
    Clean pandas series by appling a preprocessing pipeline.

    For information regarding a specific function type `help(texthero.preprocessing.func_name)`.
    The default preprocessing pipeline is the following:
    - fillna
    - lowercase
    - remove_digits
    - remove_punctuation
    - remove_diacritics
    - remove_stopwords
    - remove_whitespace
    """

    if not pipeline:
        pipeline = get_default_pipeline()

    for f in pipeline:
        s = s.pipe(f)
    return s


def has_content(s: pd.Series):
    r"""
    For each row, check that there is content.

    Examples
    --------

    >>> s = pd.Series(["c", np.nan, "\t\n", " "])
    >>> has_content(s)
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    """
    return (s.pipe(remove_whitespace) != "") & (~s.isna())


def drop_no_content(s: pd.Series):
    r"""
    Drop all rows where has_content is empty.

    Examples
    --------
    >>> s = pd.Series(["c", np.nan, "\t\n", " "])
    >>> drop_no_content(s)
    0    c
    dtype: object

    """
    return s[has_content(s)]


def remove_round_brackets(s: pd.Series):
    """
    Remove content within parentheses () and parentheses.

    Examples
    --------

    >>> s = pd.Series("Texthero (is not a superhero!)")
    >>> remove_round_brackets(s)
    0    Texthero 
    dtype: object

    """
    return s.str.replace(r"\([^()]*\)", "")


def remove_curly_brackets(s: pd.Series):
    """
    Remove content within curly brackets {} and the curly brackets.

    Examples
    -------

    >>> s = pd.Series("Texthero {is not a superhero!}")
    >>> remove_curly_brackets(s)
    0    Texthero 
    dtype: object

    """
    return s.str.replace(r"\{[^{}]*\}", "")


def remove_square_brackets(s: pd.Series):
    """
    Remove content within square brackets [] and the square brackets.

    Examples
    --------

    >>> s = pd.Series("Texthero [is not a superhero!]")
    >>> remove_square_brackets(s)
    0    Texthero 
    dtype: object

    """
    return s.str.replace(r"\[[^\[\]]*\]", "")


def remove_angle_brackets(s: pd.Series):
    """
    Remove content within angle brackets <> and the angle brackets.

    Examples
    --------

    >>> s = pd.Series("Texthero <is not a superhero!>")
    >>> remove_angle_brackets(s)
    0    Texthero 
    dtype: object

    """
    return s.str.replace(r"<[^<>]*>", "")


def remove_brackets(s: pd.Series):
    """
    Remove content within brackets and the brackets.

    Remove content from any kind of brackets, (), [], {}, <>.

    Examples
    --------

    >>> s = pd.Series("Texthero (round) [square] [curly] [angle]")
    >>> remove_brackets(s)
    0    Texthero    
    dtype: object

    See also
    --------
    remove_round_brackets(s)
    remove_curly_brackets(s)
    remove_square_brackets(s)
    remove_angle_brackets(s)
    """

    return (
        s.pipe(remove_round_brackets)
        .pipe(remove_curly_brackets)
        .pipe(remove_square_brackets)
        .pipe(remove_angle_brackets)
    )


def remove_html_tags(s: pd.Series) -> pd.Series:
    """
    Remove html tags from given input Pandas Series.

    Remove all tags of type <.*?>, such as <html>, <p>, <div class="hello">.
    Remove all html tags of type &nbsp;

    Examples
    --------
    >>> s = pd.Series("<html><h1>Title</h1></html>")
    >>> remove_html_tags(s)
    0    Title
    dtype: object
    """

    pattern = r"""(?x)                              # Turn on free-spacing
      <[^>]+>                                       # Remove <html> tags
      | &([a-z0-9]+|\#[0-9]{1,6}|\#x[0-9a-f]{1,6}); # Remove &nbsp;
      """

    return s.str.replace(pattern, "")
