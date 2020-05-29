"""
The texthero.preprocess module allow for efficient pre-processing of text-based Pandas Series and DataFrame.
"""

import re
import string
from typing import Optional, Set

import numpy as np
import pandas as pd
import unidecode
from nltk.stem import PorterStemmer, SnowballStemmer

from texthero import stopwords as _stopwords

from typing import List, Callable


def _fillna(input: pd.Series) -> pd.Series:
    """Replace not assigned values with empty spaces."""
    return input.fillna("").astype("str")


def lowercase(input: pd.Series) -> pd.Series:
    """Lowercase all text."""
    return input.str.lower()


def replace_digits(input: pd.Series, symbols: str = " ", only_blocks=True) -> pd.Series:
    """
    Replace all digits with symbols.

    By default, only replace "blocks" of digits, i.e tokens composed of only numbers.

    When `only_blocks` is set to ´False´, replace any digits.

    Parameters
    ----------
    input : Pandas Series
    symbols : str (Default single empty space " ")
        Symbols to replace
    only_blocks : bool
        When set to False, remove any digits.
    
    Returns
    -------
    Pandas Series

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("1234 falcon9")
    >>> hero.preprocessing.replace_digits(s, "X")
    0    X falcon9
    dtype: object
    >>> hero.preprocessing.replace_digits(s, "X", only_blocks=False)
    0    X falconX
    dtype: object
    """

    if only_blocks:
        pattern = r"\b\d+\b"
        return input.str.replace(pattern, symbols)
    else:
        return input.str.replace(r"\d+", symbols)


def remove_digits(input: pd.Series, only_blocks=True) -> pd.Series:
    """
    Remove all digits and replace it with a single space.

    By default, only removes "blocks" of digits. For instance, `1234 falcon9` becomes ` falcon9`.

    When the arguments `only_blocks` is set to ´False´, remove any digits.

    Parameters
    ----------
    input : Pandas Series
    only_blocks : bool
        Remove only blocks of digits.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("7ex7hero is fun 1111")
    >>> hero.preprocessing.remove_digits(s)
    0    7ex7hero is fun  
    dtype: object
    >>> hero.preprocessing.remove_digits(s, only_blocks=False)
    0     ex hero is fun  
    dtype: object
    """

    return replace_digits(input, " ", only_blocks)


def remove_punctuation(input: pd.Series) -> pd.Series:
    r"""
    Replace all punctuation with a single space (" ").

    `remove_punctuation` removes all punctuation from the given Pandas Series and replace it with a single space. Consider as punctuation characters all :data:`string.punctuation` symbols `!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).`

    See also :meth:`replace_punctuation` to replace punctuation with a custom symbol.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Finnaly.")
    >>> hero.remove_punctuation(s)
    0    Finnaly 
    dtype: object
    """
    return replace_punctuation(input, " ")


def replace_punctuation(input: pd.Series, symbol: str = " ") -> pd.Series:
    f"""
    Replace all punctuation with a given symbol.

    Parameters
    ----------
    input : Pandas Series
    symbol : str (" " by Default)
        Symbol to use as replacement for all string punctuation. 

         

    """

    return input.str.replace(rf"([{string.punctuation}])+", symbol)


def remove_diacritics(input: pd.Series) -> pd.Series:
    """
    Remove all diacritics.
    """
    return input.apply(unidecode.unidecode)


def remove_whitespace(input: pd.Series) -> pd.Series:
    """
    Remove all extra white spaces between words.
    """

    return input.str.replace("\xa0", " ").str.split().str.join(" ")


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


def get_default_pipeline() -> List[Callable[[pd.Series], pd.Series]]:
    """
    Return a list contaning all the methods used in the default cleaning pipeline.

    Return a list with the following functions:
    - _fillna
    -  lowercase
    - remove_digits
    - remove_punctuation
    - remove_diacritics
    - remove_stopwords
    - remove_whitespace
    """
    return [
        _fillna,
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
    - _fillna
    -  lowercase
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
    --------
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


    """

    return (
        s.pipe(remove_round_brackets)
        .pipe(remove_curly_brackets)
        .pipe(remove_square_brackets)
        .pipe(remove_angle_brackets)
    )


def remove_html_tags(s: pd.Series) -> pd.Series:
    """
    Remove html tags from the given Pandas Series.

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


def tokenize(s: pd.Series) -> pd.Series:
    """
    Tokenize each row of the given Series.

    Algorithm: add a space closer to a punctuation symbol at
    exception if the symbol is between two alphanumeric character and split.
    """

    pattern = (
        rf"((\w)([{string.punctuation}])(?:\B|$)|(?:^|\B)([{string.punctuation}])(\w))"
    )

    return s.str.replace(pattern, r"\2 \3 \4 \5").str.split()


def remove_urls(s: pd.Series) -> pd.Series:
    """Remove all urls from a given Series."""

    pattern = r"http\S+"

    return s.str.replace(pattern, "")
