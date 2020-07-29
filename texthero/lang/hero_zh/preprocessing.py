"""
The texthero.lang.zh.preprocessing module for Chinese.
"""

import re
import string
from typing import Optional, Set
import wrapt

import numpy as np
import pandas as pd

from spacy.lang.zh import Chinese
import texthero as hero
from texthero._helper import root_caller

from typing import List, Callable


# Ignore gensim annoying warnings
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")


def get_default_pipeline() -> List[Callable[[pd.Series], pd.Series]]:
    """
    Return a list contaning all the methods used in the default cleaning pipeline.

    Return a list with the following functions:
    1. :meth:`texthero.preprocessing.fillna`
    2. :meth:`texthero.preprocessing.remove_whitespace`
    3. :meth:`texthero.preprocessing.tokenize`
    """
    return [
        fillna,
        remove_whitespace,
        tokenize
        # remove_stopwords,     # TODO: Use a global `remove` function
    ]


def clean(s: pd.Series, pipeline=None) -> pd.Series:
    """
    Pre-process a text-based Pandas Series, by using the following default pipline.

    Default pipeline:
    1. :meth:`texthero.preprocessing.fillna`
    2. :meth:`texthero.preprocessing.remove_whitespace`
    3. :meth:`texthero.preprocessing.tokenize`
    
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

    return hero.preprocessing.clean(s, pipeline)


@root_caller(hero.preprocessing)
def fillna(s: pd.Series) -> pd.Series:
    pass


@root_caller(hero.preprocessing)
def has_content(s: pd.Series):
    pass


@root_caller(hero.preprocessing)
def drop_no_content(s: pd.Series):
    pass


@root_caller(hero.preprocessing)
def remove_html_tags(s: pd.Series) -> pd.Series:
    pass


@root_caller(hero.preprocessing)
def remove_whitespace(s: pd.Series) -> pd.Series:
    pass


@root_caller(hero.preprocessing)
def replace_urls(s: pd.Series, symbol: str) -> pd.Series:
    pass


@root_caller(hero.preprocessing)
def remove_urls(s: pd.Series) -> pd.Series:
    pass


def replace_tags(s: pd.Series, symbol: str) -> pd.Series:
    """Replace all tags from a given Pandas Series with symbol.

    A tag is a string formed by @ concatenated with a sequence of Chinese & English characters and digits. 
    Example: @我爱texthero123.

    Parameters
    ----------
    s : Pandas Series

    symbols : str
        Symbols to replace

    Examples
    --------
    >>> import texthero.lang.hero_zh as hero
    >>> import pandas as pd
    >>> s = pd.Series("你好啊@我爱texthero123。")
    >>> hero.replace_tags(s, symbol='TAG')
    0    你好啊TAG。
    dtype: object

    """

    pattern = r"@[a-zA-Z0-9\u4e00-\u9fa5]+"
    return s.str.replace(pattern, symbol)


def remove_tags(s: pd.Series) -> pd.Series:
    """Remove all tags from a given Pandas Series.

    A tag is a string formed by @ concatenated with a sequence of Chinese & English characters and digits. 
    Example: @我爱texthero123. Tags are replaced by an empty space ` `.

    Examples
    --------
    >>> import texthero.lang.hero_zh as hero
    >>> import pandas as pd
    >>> s = pd.Series("你好啊@我爱texthero123。")
    >>> hero.remove_tags(s)
    0    你好啊 。
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.replace_tags` for replacing a tag with a custom symbol.
    """
    return replace_tags(s, " ")


def replace_hashtags(s: pd.Series, symbol: str) -> pd.Series:
    """Replace all hashtags from a Pandas Series with symbol

    A hashtag is a string formed by # concatenated with a sequence of Chinese & English characters, digits and underscores. 
    Example: #杰克_texthero_123. 

    Parameters
    ----------
    s : Pandas Series

    symbols : str
        Symbols to replace
    
    Examples
    --------
    >>> import texthero.lang.hero_zh as hero
    >>> import pandas as pd
    >>> s = pd.Series("今天天气真不错#杰克_texthero_123。")
    >>> hero.replace_hashtags(s, symbol='HASHTAG')
    0    今天天气真不错HASHTAG。
    dtype: object

    """
    pattern = r"#[a-zA-Z0-9_\u4e00-\u9fa5]+"
    return s.str.replace(pattern, symbol)


def remove_hashtags(s: pd.Series) -> pd.Series:
    """Remove all hashtags from a given Pandas Series

    A hashtag is a string formed by # concatenated with a sequence of Chinese & English characters, digits and underscores. 
    Example: #杰克_texthero_123. 

    Examples
    --------
    >>> import texthero.lang.hero_zh as hero
    >>> import pandas as pd
    >>> s = pd.Series("今天天气真不错#杰克_texthero_123。")
    >>> hero.remove_hashtags(s)
    0    今天天气真不错 。
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.replace_hashtags` for replacing a hashtag with a custom symbol.
    """
    return replace_hashtags(s, " ")


def tokenize(s: pd.Series) -> pd.Series:
    """
    Tokenize each row of the given Series.

    Tokenize each row of the given Pandas Series and return a Pandas Series where each row contains a list of tokens.


    Algorithm: add a space between any punctuation symbol at
    exception if the symbol is between two alphanumeric character and split.

    Examples
    --------
    >>> import texthero.lang.hero_zh as hero
    >>> import pandas as pd
    >>> s = pd.Series(["我昨天吃烤鸭去了。"])
    >>> hero.tokenize(s)
    0    [我, 昨天, 吃, 烤鸭, 去, 了, 。]
    dtype: object

    """

    tokenizer = Chinese()
    return s.apply(lambda string: [token.text for token in tokenizer(string)])
