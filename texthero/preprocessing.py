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

    >>> import texthero
    >>> import pandas as pd
    >>> s = pd.Series(["texthero 1234 He11o"])
    >>> texthero.preprocessing.remove_digits(s)
    0    texthero He11o
    dtype: object
    >>> texthero.preprocessing.remove_digits(s, only_blocks=False)
    0    texthero   He o
    dtype: object
    """

    if only_blocks:
        return input.str.replace(r"^\d+\s|\s\d+\s|\s\d+$", " ")
    else:
        return input.str.replace(r"\d+", "")


def remove_punctuation(input : pd.Series) -> pd.Series:
    """
    Remove string.punctuation (!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).

    Replace it with a single space.
    """
    RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
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

def remove_stop_words(input: pd.Series) -> pd.Series:
    """
    Remove all stop words using NLTK stopwords list.

    List of stopwords: NLTK 'english' stopwords, 179 items.
    """
    stop_words = set(stopwords.words("english"))
    pat = r'\b(?:{})\b'.format('|'.join(stop_words))
    return input.str.replace(pat, '')




def do_stemm(input: pd.Series, stem="snowball") -> pd.Series:
    """
    Stem series using either NLTK 'porter' or 'snowball' stemmers.

    Not in the default pipeline.

    Parameters
    ----------
    input
    stem
        Can be either 'snowball' or 'stemm'

    """

    if stem is "porter":
        stemmer = PorterStemmer()
    elif stem is "snowball":
        stemmer = SnowballStemmer()
    else:
        raise ValueError("stem argument must be either 'porter' of 'stemmer'")

    def _stemm(text):
        """Stem words"""
        return " ".join([stemmer.stem(word) for word in text])

    return input.str.split().apply(_stemm)

def get_default_pipeline() -> []:
    """
    Return a list contaning all the methods used in the default cleaning pipeline.

    Return a list with the following function
     - fillna
     - lowercase
     - remove_digits
     - remove_punctuation
     - remove_diacritics
     - remove_stop_words
     - remove_whitespace
    """
    return [fillna,
            lowercase,
            remove_digits,
            remove_punctuation,
            remove_diacritics,
            remove_stop_words,
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
     - remove_stop_words
     - remove_whitespace
    """

    if not pipeline:
        pipeline = get_default_pipeline()

    for f in pipeline:
        s = s.pipe(f)
    return s

if __name__ == "__main__":
    import doctest
    doctest.testmod()
