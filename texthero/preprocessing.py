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


def fillna(input: pd.Series) -> pd.Series:
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
    symbols : str (default single empty space " ")
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

    See also :meth:`replace_digits` to replace digits with another string.

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


def replace_punctuation(input: pd.Series, symbol: str = " ") -> pd.Series:
    """
    Replace all punctuation with a given symbol.

    `replace_punctuation` replace all punctuation from the given Pandas Series and replace it with a custom symbol. It consider as punctuation characters all :data:`string.punctuation` symbols `!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).`

    Parameters
    ----------
    input : Pandas Series
    symbol : str (default single empty space)
        Symbol to use as replacement for all string punctuation. 

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Finnaly.")
    >>> hero.replace_punctuation(s, " <PUNCT> ")
    0    Finnaly <PUNCT> 
    dtype: object
    """

    return input.str.replace(rf"([{string.punctuation}])+", symbol)


def remove_punctuation(input: pd.Series) -> pd.Series:
    """
    Replace all punctuation with a single space (" ").

    `remove_punctuation` removes all punctuation from the given Pandas Series and replace it with a single space. It consider as punctuation characters all :data:`string.punctuation` symbols `!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).`

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


def remove_diacritics(input: pd.Series) -> pd.Series:
    """
    Remove all diacritics and accents.

    Remove all diacritics and accents from any word and characters from the given Pandas Series. Return a cleaned version of the Pandas Series.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Noël means Christmas in French")
    >>> hero.remove_diacritics(s)
    0    Noel means Christmas in French
    dtype: object
    """
    return input.apply(unidecode.unidecode)


def remove_whitespace(input: pd.Series) -> pd.Series:
    r"""
    Remove any extra white spaces.

    Remove any extra whitespace in the given Pandas Series. Removes also newline, tabs and any form of space.

    Useful when there is a need to visualize a Pandas Series and most cells have many newlines or other kind of space characters.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Title \n Subtitle \t    ...")
    >>> hero.remove_whitespace(s)
    0    Title Subtitle ...
    dtype: object
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

    By default uses NLTK's english stopwords of 179 words:

    Parameters
    ----------
    
    input : Pandas Series
    stopwords : Set[str], Optional
        Set of stopwords string to remove. If not passed, by default it used NLTK English stopwords. 

    Examples
    --------

    Using default NLTK list of stopwords:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero is not only for the heroes")
    >>> hero.remove_stopwords(s)
    0    Texthero      heroes
    dtype: object

    Add custom words into the default list of stopwords:

    >>> import texthero as hero
    >>> from texthero import stopwords
    >>> import pandas as pd
    >>> default_stopwords = stopwords.DEFAULT
    >>> custom_stopwords = default_stopwords.union(set(["heroes"]))
    >>> s = pd.Series("Texthero is not only for the heroes")
    >>> hero.remove_stopwords(s, custom_stopwords)
    0    Texthero      
    dtype: object


    """
    return replace_stopwords(input, symbol=" ", stopwords=stopwords)


def stem(input: pd.Series, stem="snowball", language="english") -> pd.Series:
    r"""
    Stem series using either `porter` or `snowball` NLTK stemmers.

    The act of stemming means removing the end of a words with an heuristic process. It's useful in context where the meaning of the word is important rather than his derivation. Stemming is very efficient and adapt in case the given dataset is large.

    `texthero.preprocessing.stem` make use of two NLTK stemming algorithms known as :class:`nltk.stem.SnowballStemmer` and :class:`nltk.stem.PorterStemmer`. SnowballStemmer should be used when the Pandas Series contains non-English text has it has multilanguage support.


    Parameters
    ----------
    input : Pandas Series
    stem : str (snowball by default)
        Stemming algorithm. It can be either 'snowball' or 'porter'
    language : str (english by default)
        Supported languages: `danish`, `dutch`, `english`, `finnish`, `french`, `german` , `hungarian`, `italian`, `norwegian`, `portuguese`, `romanian`, `russian`, `spanish` and `swedish`.

    Notes
    -----
    By default NLTK stemming algorithms lowercase all text.
    

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("I used to go \t\n running.")
    >>> hero.preprocessing.stem(s)
    0    i use to go running.
    dtype: object
    """

    if stem is "porter":
        stemmer = PorterStemmer()
    elif stem is "snowball":
        stemmer = SnowballStemmer(language)
    else:
        raise ValueError("stem argument must be either 'porter' of 'stemmer'")

    def _stem(text):
        return " ".join([stemmer.stem(word) for word in text])

    return input.str.split().apply(_stem)


def get_default_pipeline() -> List[Callable[[pd.Series], pd.Series]]:
    """
    Return a list contaning all the methods used in the default cleaning pipeline.

    Return a list with the following functions:
     1. :meth:`texthero.preprocessing.fillna`
     2. :meth:`texthero.preprocessing.lowercase`
     3. :meth:`texthero.preprocessing.remove_digits`
     4. :meth:`texthero.preprocessing.remove_punctuation`
     5. :meth:`texthero.preprocessing.remove_diacritics`
     6. :meth:`texthero.preprocessing.remove_stopwords`
     7. :meth:`texthero.preprocessing.remove_whitespace`
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
    Pre-process a text-based Pandas Series.

    
    Default pipeline:
     1. :meth:`texthero.preprocessing.fillna`
     2. :meth:`texthero.preprocessing.lowercase`
     3. :meth:`texthero.preprocessing.remove_digits`
     4. :meth:`texthero.preprocessing.remove_punctuation`
     5. :meth:`texthero.preprocessing.remove_diacritics`
     6. :meth:`texthero.preprocessing.remove_stopwords`
     7. :meth:`texthero.preprocessing.remove_whitespace`
    """

    if not pipeline:
        pipeline = get_default_pipeline()

    for f in pipeline:
        s = s.pipe(f)
    return s


def has_content(s: pd.Series):
    r"""
    Return a Boolean Pandas Series indicating if the rows has content.

    Examples
    --------
    >>> s = pd.Series(["content", np.nan, "\t\n", " "])
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
    Drop all rows without content.

    Drop all rows from the given Pandas Series where :meth:`has_content` is False.

    Examples
    --------
    >>> s = pd.Series(["content", np.nan, "\t\n", " "])
    >>> drop_no_content(s)
    0    content
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

    See also
    --------
    :meth:`remove_brackets`
    :meth:`remove_angle_brackets`
    :meth:`remove_curly_brackets`
    :meth:`remove_square_brackets`

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

    See also
    --------
    :meth:`remove_brackets`
    :meth:`remove_angle_brackets`
    :meth:`remove_round_brackets`
    :meth:`remove_square_brackets`

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

    See also
    --------
    :meth:`remove_brackets`
    :meth:`remove_angle_brackets`
    :meth:`remove_round_brackets`
    :meth:`remove_curly_brackets`


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

    See also
    --------
    :meth:`remove_brackets`
    :meth:`remove_round_brackets`
    :meth:`remove_curly_brackets`
    :meth:`remove_square_brackets`

    """
    return s.str.replace(r"<[^<>]*>", "")


def remove_brackets(s: pd.Series):
    """
    Remove content within brackets and the brackets itself.

    Remove content from any kind of brackets, (), [], {}, <>.

    Examples
    --------

    >>> s = pd.Series("Texthero (round) [square] [curly] [angle]")
    >>> remove_brackets(s)
    0    Texthero    
    dtype: object

    See also
    --------
    :meth:`remove_round_brackets`
    :meth:`remove_curly_brackets`
    :meth:`remove_square_brackets`
    :meth:`remove_angle_brackets`

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

    Remove all html tags of the type `<.*?>` such as <html>, <p>, <div class="hello"> and remove all html tags of type &nbsp and return a cleaned Pandas Series.

    Examples
    --------
    >>> s = pd.Series("<html><h1>Title</h1></html>")
    >>> remove_html_tags(s)
    0    Title
    dtype: object
    
    """

    pattern = r"""(?x)                    # Turn on free-spacing
      <[^>]+>                             # Remove <html> tags
      | &([a-z0-9]+|\#[0-9]{1,6}|\#x[0-9a-f]{1,6}); # Remove &nbsp;
      """

    return s.str.replace(pattern, "")


def tokenize(s: pd.Series) -> pd.Series:
    """
    Tokenize each row of the given Series.

    Tokenize each row of the given Pandas Series and return a Pandas Series where each row contains a list of tokens.


    Algorithm: add a space between any punctuation symbol at
    exception if the symbol is between two alphanumeric character and split.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Today you're looking great!"])
    >>> hero.tokenize(s)
    0    [Today, you're, looking, great, !]
    dtype: object

    """

    pattern = (
        rf"((\w)([{string.punctuation}])(?:\B|$)|(?:^|\B)([{string.punctuation}])(\w))"
    )

    return s.str.replace(pattern, r"\2 \3 \4 \5").str.split()


def replace_urls(s: pd.Series, symbol: str) -> pd.Series:
    r"""Replace all urls with the given symbol.

    `replace_urls` replace any urls from the given Pandas Series with the given symbol.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Go to: https://example.com")
    >>> hero.replace_urls(s, "<URL>")
    0    Go to: <URL>
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.remove_urls`

    """

    pattern = r"http\S+"

    return s.str.replace(pattern, symbol)


def remove_urls(s: pd.Series) -> pd.Series:
    r"""Remove all urls from a given Pandas Series.

    `remove_urls` remove any urls and replace it with a single empty space.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Go to: https://example.com")
    >>> hero.remove_urls(s)
    0    Go to:  
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.replace_urls`

    """

    return replace_urls(s, " ")
