"""
The texthero.preprocess module allow for efficient pre-processing of text-based Pandas Series and DataFrame.
"""

from gensim.sklearn_api.phrases import PhrasesTransformer
import re
import string
from typing import Optional, Set
import unicodedata

import numpy as np
import pandas as pd
import unidecode
from nltk.stem import PorterStemmer, SnowballStemmer

from texthero import stopwords as _stopwords

from typing import List, Callable

# Ignore gensim annoying warnings
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")


def fillna(s: pd.Series) -> pd.Series:
    """
    Replaces not assigned values with empty spaces.


    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series([np.NaN, "I'm", "You're"])
    >>> hero.fillna(s)
    0          
    1       I'm
    2    You're
    dtype: object
    """
    return s.fillna("").astype("str")


def lowercase(s: pd.Series) -> pd.Series:
    """
    Lowercase all texts in a series.

    
    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("This is NeW YoRk wIth upPer letters")
    >>> hero.lowercase(s)
    0    this is new york with upper letters
    dtype: object
    """
    return s.str.lower()


def replace_digits(s: pd.Series, symbols: str = " ", only_blocks=True) -> pd.Series:
    """
    Replace all digits with symbols.

    By default, only replaces "blocks" of digits, i.e tokens composed of only numbers.

    When `only_blocks` is set to ´False´, replaces all digits.

    Parameters
    ----------
    s : Pandas Series

    symbols : str (default single empty space " ")
        Symbols to replace

    only_blocks : bool
        When set to False, replace all digits.

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
        return s.str.replace(pattern, symbols)
    else:
        return s.str.replace(r"\d+", symbols)


def remove_digits(s: pd.Series, only_blocks=True) -> pd.Series:
    """
    Remove all digits and replaces them with a single space.

    By default, only remove "blocks" of digits. For instance, `1234 falcon9` becomes ` falcon9`.

    When the arguments `only_blocks` is set to ´False´, remove any digits.

    See also :meth:`replace_digits` to replace digits with another string.

    Parameters
    ----------
    s : Pandas Series

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

    return replace_digits(s, " ", only_blocks)


def replace_punctuation(s: pd.Series, symbol: str = " ") -> pd.Series:
    """
    Replace all punctuation with a given symbol.

    Replace all punctuation from the given
    Pandas Series with a custom symbol. 
    It considers as punctuation characters all :data:`string.punctuation` 
    symbols `!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).`


    Parameters
    ----------
    s : Pandas Series

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

    return s.str.replace(rf"([{string.punctuation}])+", symbol)


def remove_punctuation(s: pd.Series) -> pd.Series:
    """
    Replace all punctuation with a single space (" ").

    Remove all punctuation from the given Pandas Series and replace it
    with a single space. It considers as punctuation characters all :data:`string.punctuation` symbols `!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).`

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
    return replace_punctuation(s, " ")


def _remove_diacritics(text: str) -> str:
    """
    Remove diacritics and accents from one string.

    Examples
    --------
    >>> from texthero.preprocessing import _remove_diacritics
    >>> import pandas as pd
    >>> text = "Montréal, über, 12.89, Mère, Françoise, noël, 889, اِس, اُس"
    >>> _remove_diacritics(text)
    'Montreal, uber, 12.89, Mere, Francoise, noel, 889, اس, اس'
    """
    nfkd_form = unicodedata.normalize("NFKD", text)
    # unicodedata.combining(char) checks if the character is in
    # composed form (consisting of several unicode chars combined), i.e. a diacritic
    return "".join([char for char in nfkd_form if not unicodedata.combining(char)])


def remove_diacritics(s: pd.Series) -> pd.Series:
    """
    Remove all diacritics and accents.

    Remove all diacritics and accents from any word and characters from the given Pandas Series.
    Return a cleaned version of the Pandas Series.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Montréal, über, 12.89, Mère, Françoise, noël, 889, اِس, اُس")
    >>> hero.remove_diacritics(s)[0]
    'Montreal, uber, 12.89, Mere, Francoise, noel, 889, اس, اس'

    """
    return s.astype("unicode").apply(_remove_diacritics)


def remove_whitespace(s: pd.Series) -> pd.Series:
    r"""
    Remove any extra white spaces.

    Remove any extra whitespace in the given Pandas Series.
    Remove also newline, tabs and any form of space.

    Useful when there is a need to visualize a Pandas Series and
    most cells have many newlines or other kind of space characters.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Title \n Subtitle \t    ...")
    >>> hero.remove_whitespace(s)
    0    Title Subtitle ...
    dtype: object
    """

    return s.str.replace("\xa0", " ").str.split().str.join(" ")


def _replace_stopwords(text: str, words: Set[str], symbol: str = " ") -> str:
    """
    Remove words in a set from a string, replacing them with a symbol.

    Parameters
    ----------
    text: str

    stopwords : Set[str]
        Set of stopwords string to remove.

    symbol: str, Optional
        Character(s) to replace words with; defaults to a space.

    Examples
    --------
    >>> from texthero.preprocessing import _replace_stopwords
    >>> s = "the book of the jungle"
    >>> symbol = "$"
    >>> stopwords = ["the", "of"]
    >>> _replace_stopwords(s, stopwords, symbol)
    '$ book $ $ jungle'

    """

    pattern = r"""(?x)                          # Set flag to allow verbose regexps
      \w+(?:-\w+)*                              # Words with optional internal hyphens 
      | \s*                                     # Any space
      | [][!"#$%&'*+,-./:;<=>?@\\^():_`{|}~]    # Any symbol 
    """

    return "".join(t if t not in words else symbol for t in re.findall(pattern, text))


def replace_stopwords(
    s: pd.Series, symbol: str, stopwords: Optional[Set[str]] = None
) -> pd.Series:
    """
    Replace all instances of `words` with symbol.

    By default uses NLTK's english stopwords of 179 words.

    Parameters
    ----------
    s : Pandas Series

    symbol: str
        Character(s) to replace words with.

    stopwords : Set[str], Optional
        Set of stopwords string to remove. If not passed, by default it used NLTK English stopwords. 

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("the book of the jungle")
    >>> hero.replace_stopwords(s, "X")
    0    X book X X jungle
    dtype: object

    """

    if stopwords is None:
        stopwords = _stopwords.DEFAULT
    return s.apply(_replace_stopwords, args=(stopwords, symbol))


def remove_stopwords(
    s: pd.Series, stopwords: Optional[Set[str]] = None, remove_str_numbers=False
) -> pd.Series:
    """
    Remove all instances of `words`.

    By default use NLTK's english stopwords of 179 words:

    Parameters
    ----------
    s : Pandas Series

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
    return replace_stopwords(s, symbol="", stopwords=stopwords)


def stem(s: pd.Series, stem="snowball", language="english") -> pd.Series:
    r"""
    Stem series using either `porter` or `snowball` NLTK stemmers.

    The act of stemming means removing the end of a words with an heuristic process.
    It's useful in context where the meaning of the word is important rather than his derivation. Stemming is very efficient and adapt in case the given dataset is large.

    Make use of two NLTK stemming algorithms known as :class:`nltk.stem.SnowballStemmer` and :class:`nltk.stem.PorterStemmer`. SnowballStemmer should be used when the Pandas Series contains non-English text has it has multilanguage support.


    Parameters
    ----------
    s : Pandas Series

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
    >>> hero.stem(s)
    0    i use to go running.
    dtype: object
    """

    if stem == "porter":
        stemmer = PorterStemmer()
    elif stem == "snowball":
        stemmer = SnowballStemmer(language)
    else:
        raise ValueError("stem argument must be either 'porter' of 'stemmer'")

    def _stem(text):
        return " ".join([stemmer.stem(word) for word in text])

    return s.str.split().apply(_stem)


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
    Pre-process a text-based Pandas Series, by using the following default pipline.

     Default pipeline:
     1. :meth:`texthero.preprocessing.fillna`
     2. :meth:`texthero.preprocessing.lowercase`
     3. :meth:`texthero.preprocessing.remove_digits`
     4. :meth:`texthero.preprocessing.remove_punctuation`
     5. :meth:`texthero.preprocessing.remove_diacritics`
     6. :meth:`texthero.preprocessing.remove_stopwords`
     7. :meth:`texthero.preprocessing.remove_whitespace`

    Parameters
    ----------
    s : Pandas Series

    pipeline :List[Callable[[Pandas Series], Pandas Series]]
       inserting specific pipeline to clean a text
   
    Examples
    --------
    For the default pipeline:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Uper 9dig.        he her ÄÖÜ")
    >>> hero.clean(s)
    0    uper 9dig aou
    dtype: object
    """

    if not pipeline:
        pipeline = get_default_pipeline()

    for f in pipeline:
        s = s.pipe(f)
    return s


def has_content(s: pd.Series) -> pd.Series:
    r"""
    Return a Boolean Pandas Series indicating if the rows have content.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["content", np.nan, "\t\n", " "])
    >>> hero.has_content(s)
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    """
    return (s.pipe(remove_whitespace) != "") & (~s.isna())


def drop_no_content(s: pd.Series) -> pd.Series:
    r"""
    Drop all rows without content.

    Every row from a given Pandas Series, where :meth:`has_content` is False, will be dropped.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["content", np.nan, "\t\n", " "])
    >>> hero.drop_no_content(s)
    0    content
    dtype: object

    """
    return s[has_content(s)]


def remove_round_brackets(s: pd.Series) -> pd.Series:
    """
    Remove content within parentheses '()' and the parentheses by themself.

    Examples
    --------

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero (is not a superhero!)")
    >>> hero.remove_round_brackets(s)
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


def remove_curly_brackets(s: pd.Series) -> pd.Series:
    """
    Remove content within curly brackets '{}' and the curly brackets by themself.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero {is not a superhero!}")
    >>> hero.remove_curly_brackets(s)
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


def remove_square_brackets(s: pd.Series) -> pd.Series:
    """
    Remove content within square brackets '[]' and the square brackets by themself.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero [is not a superhero!]")
    >>> hero.remove_square_brackets(s)
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


def remove_angle_brackets(s: pd.Series) -> pd.Series:
    """
    Remove content within angle brackets '<>' and the angle brackets by themself.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero <is not a superhero!>")
    >>> hero.remove_angle_brackets(s)
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


def remove_brackets(s: pd.Series) -> pd.Series:
    """
    Remove content within brackets and the brackets itself.

    Remove content from any kind of brackets, (), [], {}, <>.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Texthero (round) [square] [curly] [angle]")
    >>> hero.remove_brackets(s)
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

    Remove all html tags of the type `<.*?>` such as <html>, <p>, <div class="hello"> and
    remove all html tags of type &nbsp and return a cleaned Pandas Series.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("<html><h1>Title</h1></html>")
    >>> hero.remove_html_tags(s)
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

    Tokenize each row of the given Pandas Series and return a Pandas Series where
    each row contains a list of tokens.

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

    punct = string.punctuation.replace("_", "")
    # In regex, the metacharacter 'w' is "a-z, A-Z, 0-9, including the _ (underscore) character." We therefore remove it from the punctuation string as this is already included in \w

    pattern = rf"((\w)([{punct}])(?:\B|$)|(?:^|\B)([{punct}])(\w))"

    return s.str.replace(pattern, r"\2 \3 \4 \5").str.split()


# Warning message for not-tokenized inputs
_not_tokenized_warning_message = (
    "It seems like the given Pandas Series s is not tokenized. This function will"
    " tokenize it automatically using hero.tokenize(s) first. You should consider"
    " tokenizing it yourself first with hero.tokenize(s) in the future."
)


def phrases(s: pd.Series, min_count: int = 5, threshold: int = 10, symbol: str = "_"):
    r"""Group up collocations words

    Given a pandas Series of tokenized strings, group together bigrams where
    each tokens has at least `min_count` term frequency and where the
    `threshold` is larger than the underline formula.

    :math:`\frac{(bigram\_a\_b\_count - min\_count)* len\_vocab }
    { (word\_a\_count * word\_b\_count)}`.

    Parameters
    ----------
    s : Pandas Series
    
    min_count : Int, optional. Default is 5.
        ignore tokens with frequency less than this
        
    threshold : Int, optional. Default is 10.
        ignore tokens with a score under that threshold
        
    symbol : Str, optional. Default is '_'.
        character used to join collocation words

    Examples
    --------
    >>> import texthero as hero
    >>> s = pd.Series([['New', 'York', 'is', 'a', 'beautiful', 'city'],
    ...               ['Look', ':', 'New', 'York', '!']])
    >>> hero.phrases(s, min_count=1, threshold=1)
    0    [New_York, is, a, beautiful, city]
    1                [Look, :, New_York, !]
    dtype: object

    Reference
    --------
    `Mikolov, et. al: "Distributed Representations of Words and Phrases and
    their Compositionality"
        <https://arxiv.org/abs/1310.4546>`_

    """

    if not isinstance(s.iloc[0], list):
        warnings.warn(_not_tokenized_warning_message, DeprecationWarning)
        s = tokenize(s)

    delimiter = symbol.encode("utf-8")
    phrases = PhrasesTransformer(
        min_count=min_count, threshold=threshold, delimiter=delimiter
    )
    return pd.Series(phrases.fit_transform(s.values), index=s.index)


def replace_urls(s: pd.Series, symbol: str) -> pd.Series:
    r"""Replace all urls with the given symbol.

    Replace any urls from the given Pandas Series with the given symbol.

    Parameters
    ----------
    s: Pandas Series

    symbol: String
        The symbol to which the URL should be changed to.

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

    Remove all urls and replaces them with a single empty space.

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


def replace_tags(s: pd.Series, symbol: str) -> pd.Series:
    """Replace all tags from a given Pandas Series with symbol.

    A tag is a string formed by @ concatenated with a sequence of characters and digits. Example: @texthero123.

    Parameters
    ----------
    s : Pandas Series

    symbols : str
        Symbols to replace

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Hi @texthero123, we will replace you")
    >>> hero.replace_tags(s, symbol='TAG')
    0    Hi TAG, we will replace you
    dtype: object

    """

    pattern = r"@[a-zA-Z0-9]+"
    return s.str.replace(pattern, symbol)


def remove_tags(s: pd.Series) -> pd.Series:
    """Remove all tags from a given Pandas Series.

    A tag is a string formed by @ concatenated with a sequence of characters and digits. Example: @texthero123. Tags are replaceb by an empty space ` `.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Hi @tag, we will remove you")
    >>> hero.remove_tags(s)
    0    Hi  , we will remove you
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.replace_tags` for replacing a tag with a custom symbol.
    """
    return replace_tags(s, " ")


def replace_hashtags(s: pd.Series, symbol: str) -> pd.Series:
    """Replace all hashtags from a Pandas Series with symbol

    A hashtag is a string formed by # concatenated with a sequence of characters, digits and underscores. Example: #texthero_123. 

    Parameters
    ----------
    s : Pandas Series

    symbols : str
        Symbols to replace
    
    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Hi #texthero_123, we will replace you.")
    >>> hero.replace_hashtags(s, symbol='HASHTAG')
    0    Hi HASHTAG, we will replace you.
    dtype: object

    """
    pattern = r"#[a-zA-Z0-9_]+"
    return s.str.replace(pattern, symbol)


def remove_hashtags(s: pd.Series) -> pd.Series:
    """Remove all hashtags from a given Pandas Series

    A hashtag is a string formed by # concatenated with a sequence of characters, digits and underscores. Example: #texthero_123. 

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Hi #texthero_123, we will remove you.")
    >>> hero.remove_hashtags(s)
    0    Hi  , we will remove you.
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.replace_hashtags` for replacing a hashtag with a custom symbol.
    """
    return replace_hashtags(s, " ")
