"""
Preprocess text-based Pandas DataFrame.
"""

import re
import string
import unidecode
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
import pandas as pd
import numpy as np


def fillna(input: pd.Series) -> pd.Series:
    """Replace not assigned values with empty spaces."""
    return input.fillna("").astype("str")


def lowercase(input: pd.Series) -> pd.Series:
    """Lowercase all text."""
    return input.str.lower()


def remove_digits(input: pd.Series, only_blocks=True) -> pd.Series:
    """
    Remove all digits from a series and replace it with a single space.

    Parameters
    ----------

    input : pd.Series
    only_blocks : bool
                  Remove only blocks of digits. For instance, `hel1234lo 1234` becomes `hel1234lo`.

    Examples
    --------
    >>> s = pd.Series("7ex7hero is fun 1111")
    >>> remove_digits(s)
    0    7ex7hero is fun 
    dtype: object
    >>> remove_digits(s, only_blocks=False)
    0    exhero is fun 
    dtype: object
    """

    if only_blocks:
        return input.str.replace(r"^\d+\s|\s\d+\s|\s\d+$", " ")
    else:
        return input.str.replace(r"\d+", "")


def remove_punctuation(input: pd.Series) -> pd.Series:
    """
    Remove string.punctuation (!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).

    Replace it with a single space.
    """
    RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation),
                          re.UNICODE)
    return input.str.replace(RE_PUNCT, " ")


def remove_diacritics(input: pd.Series) -> pd.Series:
    """
    Remove all diacritics.
    """
    return input.apply(unidecode.unidecode)


def remove_whitespace(input: pd.Series) -> pd.Series:
    """
    Remove all white spaces between words.
    """

    return input.str.replace(u"\xa0", u" ").str.split().str.join(" ")


def remove_stopwords(input: pd.Series) -> pd.Series:
    """
    Remove all stop words using NLTK stopwords list.

    List of stopwords: NLTK 'english' stopwords, 179 items.
    """
    stop_words = set(stopwords.words("english"))
    pat = r'\b(?:{})\b'.format('|'.join(stop_words))
    return input.str.replace(pat, '')


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
        Supportted languages: 
            danish dutch english finnish french german hungarian italian 
            norwegian porter portuguese romanian russian spanish swedish
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

    Return a list with the following function
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
    """
    For each row, check that there is content.

    Example
    -------

    >>> s = pd.Series(["c", np.nan, "\t\\n", " "])
    >>> has_content(s)
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    """
    return (s.pipe(remove_whitespace) != "") & (~s.isna())


def drop_no_content(s: pd.Series):
    """
    Drop all rows where has_content is empty.

    Example
    -------

    >>> s = pd.Series(["c", np.nan, "\t\\n", " "])
    >>> drop_no_content(s)
    0    c
    dtype: object

    """
    return s[has_content(s)]


def remove_round_brackets(s: pd.Series):
    """
    Remove content within parentheses () and parentheses.

    Example
    -------

    >>> s = pd.Series("Texthero (is not a superhero!)")
    >>> remove_round_brackets(s)
    0    Texthero 
    dtype: object

    """
    return s.str.replace(r"\([^()]*\)", "")


def remove_curly_brackets(s: pd.Series):
    """
    Remove content within curly brackets {} and the curly brackets.

    Example
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

    Example
    -------

    >>> s = pd.Series("Texthero [is not a superhero!]")
    >>> remove_square_brackets(s)
    0    Texthero 
    dtype: object

    """
    return s.str.replace(r"\[[^\[\]]*\]", "")


def remove_angle_brackets(s: pd.Series):
    """
    Remove content within angle brackets <> and the angle brackets.

    Example
    -------

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

    Example
    -------

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
    return (s.pipe(remove_round_brackets).pipe(remove_curly_brackets).pipe(
        remove_square_brackets).pipe(remove_angle_brackets))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
