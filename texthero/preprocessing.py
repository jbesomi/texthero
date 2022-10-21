"""
The texthero.preprocess module allow for efficient pre-processing of text-based Pandas Series and DataFrame.
"""

from gensim.models.phrases import Phrases
import re
import string
from typing import Optional, Set
import unicodedata

import numpy as np
import pandas as pd

from texthero._types import TokenSeries, TextSeries, InputSeries

from typing import List, Callable, Union

# Ignore gensim annoying warnings
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")


@InputSeries(TextSeries)
def fillna(s: TextSeries, replace_string="") -> TextSeries:
    """
    Replaces not assigned values with empty or given string.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> import numpy as np
    >>> s = pd.Series(["I'm", np.NaN, pd.NA, "You're"])
    >>> hero.fillna(s)
    0       I'm
    1          
    2          
    3    You're
    dtype: object
    >>> hero.fillna(s, "Missing")
    0        I'm
    1    Missing
    2    Missing
    3     You're
    dtype: object
    """

    return s.fillna(replace_string).astype("str")


@InputSeries(TextSeries)
def lowercase(s: TextSeries) -> TextSeries:
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


@InputSeries(TextSeries)
def replace_digits(s: TextSeries, symbols: str = " ", only_blocks=True) -> TextSeries:
    """
    Replace all digits with symbols.

    By default, only replaces "blocks" of digits, i.e tokens composed of only
    numbers.

    When `only_blocks` is set to ´False´, replaces all digits.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbols : str, optional, default=" "
        Symbols to replace

    only_blocks : bool, optional, default=True
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


@InputSeries(TextSeries)
def remove_digits(s: TextSeries, only_blocks=True) -> TextSeries:
    """
    Remove all digits and replaces them with a single space.

    By default, only remove "blocks" of digits. For instance, `1234 falcon9`
    becomes ` falcon9`.

    When the arguments `only_blocks` is set to ´False´, remove any digits.

    See also :meth:`replace_digits` to replace digits with another string.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    only_blocks : bool, optional, default=True
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


@InputSeries(TextSeries)
def replace_punctuation(s: TextSeries, symbol: str = " ") -> TextSeries:
    """
    Replace all punctuation with a given symbol.

    Replace all punctuation from the given
    Pandas Series with a custom symbol. 
    It considers as punctuation characters all :data:`string.punctuation` 
    symbols `!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).`


    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbol : str, optional, default=" "
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


@InputSeries(TextSeries)
def remove_punctuation(s: TextSeries) -> TextSeries:
    """
    Replace all punctuation with a single space (" ").

    Remove all punctuation from the given Pandas Series and replace it
    with a single space. It considers as punctuation characters all
    :data:`string.punctuation` symbols `!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~).`

    See also :meth:`replace_punctuation` to replace punctuation with a custom
    symbol.

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


@InputSeries(TextSeries)
def remove_diacritics(s: TextSeries) -> TextSeries:
    """
    Remove all diacritics and accents.

    Remove all diacritics and accents from any word and characters from the
    given Pandas Series.
    Return a cleaned version of the Pandas Series.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(
    ...     "Montréal, über, 12.89, Mère, Françoise, noël, 889, اِس, اُس")
    >>> hero.remove_diacritics(s)[0]
    'Montreal, uber, 12.89, Mere, Francoise, noel, 889, اس, اس'

    """
    return s.astype("unicode").apply(_remove_diacritics)


@InputSeries(TextSeries)
def remove_whitespace(s: TextSeries) -> TextSeries:
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

    symbol: str, optional, default=" "
        Character(s) to replace words with.

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


@InputSeries(TextSeries)
def replace_stopwords(
    s: TextSeries, symbol: str, stopwords: Optional[Set[str]] = None
) -> TextSeries:
    """
    Replace all instances of `words` with symbol.

    By default uses NLTK's english stopwords of 179 words.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbol: str
        Character(s) to replace words with.

    stopwords : Set[str], optional, default=None
        Set of stopwords string to remove. If not passed,
        by default uses NLTK English stopwords. 

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
        from texthero import stopwords as _stopwords

        stopwords = _stopwords.DEFAULT
    return s.apply(_replace_stopwords, args=(stopwords, symbol))


@InputSeries(TextSeries)
def remove_stopwords(
    s: TextSeries, stopwords: Optional[Set[str]] = None, remove_str_numbers=False
) -> TextSeries:
    """
    Remove all instances of `words`.

    By default use NLTK's english stopwords of 179 words:

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    stopwords : Set[str], optional, default=None
        Set of stopwords string to remove. If not passed,
        by default uses NLTK English stopwords.

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


def get_default_pipeline() -> List[Callable[[pd.Series], pd.Series]]:
    """
    Return a list contaning all the methods used in the default cleaning
    pipeline.

    Return a list with the following functions:
     1. :meth:`texthero.preprocessing.fillna`
     2. :meth:`texthero.preprocessing.lowercase`
     3. :meth:`texthero.preprocessing.remove_digits`
     4. :meth:`texthero.preprocessing.remove_html_tags`
     5. :meth:`texthero.preprocessing.remove_punctuation`
     6. :meth:`texthero.preprocessing.remove_diacritics`
     7. :meth:`texthero.preprocessing.remove_stopwords`
     8. :meth:`texthero.preprocessing.remove_whitespace`
    """
    return [
        fillna,
        lowercase,
        remove_digits,
        remove_html_tags,
        remove_punctuation,
        remove_diacritics,
        remove_stopwords,
        remove_whitespace,
    ]


@InputSeries(TextSeries)
def clean(s: TextSeries, pipeline=None) -> TextSeries:
    """
    Pre-process a text-based Pandas Series, by using the following default
    pipeline.

     Default pipeline:
     1. :meth:`texthero.preprocessing.fillna`
     2. :meth:`texthero.preprocessing.lowercase`
     3. :meth:`texthero.preprocessing.remove_digits`
     4. :meth:`texthero.preprocessing.remove_html_tags`
     5. :meth:`texthero.preprocessing.remove_punctuation`
     6. :meth:`texthero.preprocessing.remove_diacritics`
     7. :meth:`texthero.preprocessing.remove_stopwords`
     8. :meth:`texthero.preprocessing.remove_whitespace`

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    pipeline : List[Callable[Pandas Series, Pandas Series]],
               optional, default=None
       Specific pipeline to clean the texts. Has to be a list
       of functions taking as input and returning as output
       a Pandas Series. If None, the default pipeline
       is used.
   
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


@InputSeries(TextSeries)
def has_content(s: TextSeries) -> TextSeries:
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


@InputSeries(TextSeries)
def drop_no_content(s: TextSeries) -> TextSeries:
    r"""
    Drop all rows without content.

    Every row from a given Pandas Series, where :meth:`has_content` is False,
    will be dropped.

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


@InputSeries(TextSeries)
def remove_round_brackets(s: TextSeries) -> TextSeries:
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


@InputSeries(TextSeries)
def remove_curly_brackets(s: TextSeries) -> TextSeries:
    """
    Remove content within curly brackets '{}' and the curly brackets by
    themselves.

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


@InputSeries(TextSeries)
def remove_square_brackets(s: TextSeries) -> TextSeries:
    """
    Remove content within square brackets '[]' and the square brackets by
    themselves.

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


@InputSeries(TextSeries)
def remove_angle_brackets(s: TextSeries) -> TextSeries:
    """
    Remove content within angle brackets '<>' and the angle brackets by
    themselves.

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


@InputSeries(TextSeries)
def remove_brackets(s: TextSeries) -> TextSeries:
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


@InputSeries(TextSeries)
def remove_html_tags(s: TextSeries) -> TextSeries:
    """
    Remove html tags from the given Pandas Series.

    Remove all html tags of the type `<.*?>` such as <html>, <p>,
    <div class="hello"> and remove all html tags of type &nbsp and return a
    cleaned Pandas Series.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("<html><h1>Title</h1></html>")
    >>> hero.remove_html_tags(s)
    0    Title
    dtype: object

    """

    pattern = r"""(?x)                              # Turn on free-spacing
      <[^>]+>                                       # Remove <html> tags
      | &([a-z0-9]+|\#[0-9]{1,6}|\#x[0-9a-f]{1,6}); # Remove &nbsp;
      """

    return s.str.replace(pattern, "")


@InputSeries(TextSeries)
def tokenize(s: TextSeries) -> TokenSeries:
    """
    Tokenize each row of the given Series.

    Tokenize each row of the given Pandas Series and return a Pandas Series
    where each row contains a list of tokens.

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
    # In regex, the metacharacter 'w' is "a-z, A-Z, 0-9, including the _ (underscore)
    # character." We therefore remove it from the punctuation string as this is already
    # included in \w.

    pattern = rf"((\w)([{punct}])(?:\B|$)|(?:^|\B)([{punct}])(\w))"

    return s.str.replace(pattern, r"\2 \3 \4 \5").str.split()


# Warning message for not-tokenized inputs
_not_tokenized_warning_message = (
    "It seems like the given Pandas Series s is not tokenized. This function will"
    " tokenize it automatically using hero.tokenize(s) first. You should consider"
    " tokenizing it yourself first with hero.tokenize(s) in the future."
)


def phrases(
    s: TokenSeries, min_count: int = 5, threshold: int = 10, symbol: str = "_"
) -> TokenSeries:
    r"""Group up collocations words

    Given a pandas Series of tokenized strings, group together bigrams where
    each tokens has at least `min_count` term frequency and where the
    `threshold` is larger than the underline formula.

    :math:`\frac{(bigram\_a\_b\_count - min\_count)* len\_vocab }
    { (word\_a\_count * word\_b\_count)}`.

    Parameters
    ----------
    s : :class:`texthero._types.TokenSeries`
    
    min_count : int, optional, default=5
        Ignore tokens with frequency less than this.
        
    threshold : int, optional, default=10
        Ignore tokens with a score under that threshold.
        
    symbol : str, optional, default="_"
        Character used to join collocation words.

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
    phrases_model = Phrases(
        sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS
    )
    output = pd.Series(phrases_model[s.values], index=s.index)
    print(output)
    return output 


@InputSeries(TextSeries)
def replace_urls(s: TextSeries, symbol: str) -> TextSeries:
    r"""Replace all urls with the given symbol.

    Replace any urls from the given Pandas Series with the given symbol.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbol : str
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


@InputSeries(TextSeries)
def remove_urls(s: TextSeries) -> TextSeries:
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


@InputSeries(TextSeries)
def replace_tags(s: TextSeries, symbol: str) -> TextSeries:
    """Replace all tags from a given Pandas Series with symbol.

    A tag is a string formed by @ concatenated with a sequence of characters
    and digits. Example: @texthero123.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbols : str
        Symbol to replace tags with.

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


@InputSeries(TextSeries)
def remove_tags(s: TextSeries) -> TextSeries:

    """Remove all tags from a given Pandas Series.

    A tag is a string formed by @ concatenated with a sequence of characters
    and digits. Example: @texthero123. Tags are replaceb by an empty space ` `.

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
    :meth:`texthero.preprocessing.replace_tags` for replacing a tag with a
        custom symbol.
    """
    return replace_tags(s, " ")


@InputSeries(TextSeries)
def replace_hashtags(s: TextSeries, symbol: str) -> TextSeries:
    """Replace all hashtags from a Pandas Series with symbol

    A hashtag is a string formed by # concatenated with a sequence of
    characters, digits and underscores. Example: #texthero_123. 

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbol : str
        Symbol to replace hashtags with.
    
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


@InputSeries(TextSeries)
def remove_hashtags(s: TextSeries) -> TextSeries:
    """Remove all hashtags from a given Pandas Series

    A hashtag is a string formed by # concatenated with a sequence of
    characters, digits and underscores. Example: #texthero_123. 

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
    :meth:`texthero.preprocessing.replace_hashtags` for replacing a hashtag
        with a custom symbol.
    """
    return replace_hashtags(s, " ")
