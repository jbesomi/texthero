"""
The texthero.lang.zh.preprocessing module for Chinese.
"""

import re
import string
from typing import Optional, Set

import numpy as np
import pandas as pd

from spacy.lang.zh import Chinese

import texthero as hero
from texthero._types import TokenSeries, TextSeries, InputSeries

# Standard functions that supports Chinese
from texthero.preprocessing import (
    fillna,
    has_content,
    drop_no_content,
    remove_whitespace,
    remove_html_tags,
    replace_urls,
    remove_urls,
    phrases,
)

from typing import List, Callable


# Ignore gensim annoying warnings
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")


__all__ = [
    "fillna",
    "has_content",
    "drop_no_content",
    "remove_whitespace",
    "remove_html_tags",
    "replace_urls",
    "remove_urls",
    "phrases",
    "clean",
    "get_default_pipeline",
    "remove_hashtags",
    "remove_tags",
    "replace_hashtags",
    "replace_tags",
    "tokenize",
]


def get_default_pipeline() -> List[Callable[[pd.Series], pd.Series]]:
    """
    Return a list with the following functions:
    1. :meth:`texthero.preprocessing.fillna`
    2. :meth:`texthero.preprocessing.remove_whitespace`
    3. :meth:`texthero.preprocessing.tokenize`

    See also
    --------
    :meth:`texthero.preprocessing.get_default_pipeline`
    """
    return [
        fillna,
        remove_whitespace,
        tokenize
        # remove_stopwords,     # TODO: Use a global `remove` function
    ]


@InputSeries(TextSeries)
def clean(s: TextSeries, pipeline=None) -> TextSeries:
    """
    Default pipeline:
    1. :meth:`texthero.preprocessing.fillna`
    2. :meth:`texthero.preprocessing.remove_whitespace`
    3. :meth:`texthero.preprocessing.tokenize`
    
    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    pipeline :List[Callable[[Pandas Series], Pandas Series]]
       inserting specific pipeline to clean a text
   
    Examples
    --------
    For the default pipeline:

    >>> import texthero.lang.zh as hero
    >>> import pandas as pd
    >>> s = pd.Series("我昨天吃烤鸭去了。     挺好吃的。")
    >>> hero.clean(s)
    0    [我, 昨天, 吃, 烤鸭, 去, 了, 。, 挺好吃, 的, 。]
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.clean`
    """
    if not pipeline:
        pipeline = get_default_pipeline()

    return hero.preprocessing.clean(s, pipeline)


@InputSeries(TextSeries)
def replace_tags(s: TextSeries, symbol: str) -> TextSeries:
    """
    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbols : str
        Symbols to replace

    Examples
    --------
    >>> import texthero.lang.zh as hero
    >>> import pandas as pd
    >>> s = pd.Series("你好啊@我爱texthero123。")
    >>> hero.replace_tags(s, symbol='TAG')
    0    你好啊TAG。
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.replace_tags`
    """

    pattern = r"@[a-zA-Z0-9\u4e00-\u9fa5]+"
    return s.str.replace(pattern, symbol)


@InputSeries(TextSeries)
def remove_tags(s: TextSeries) -> TextSeries:
    """
    Examples
    --------
    >>> import texthero.lang.zh as hero
    >>> import pandas as pd
    >>> s = pd.Series("你好啊@我爱texthero123。")
    >>> hero.remove_tags(s)
    0    你好啊 。
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.remove_tags`
    """
    return replace_tags(s, " ")


@InputSeries(TextSeries)
def replace_hashtags(s: TextSeries, symbol: str) -> TextSeries:
    """
    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    symbols : str
        Symbols to replace
    
    Examples
    --------
    >>> import texthero.lang.zh as hero
    >>> import pandas as pd
    >>> s = pd.Series("今天天气真不错#杰克_texthero_123。")
    >>> hero.replace_hashtags(s, symbol='HASHTAG')
    0    今天天气真不错HASHTAG。
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.replace_hashtags`
    """
    pattern = r"#[a-zA-Z0-9_\u4e00-\u9fa5]+"
    return s.str.replace(pattern, symbol)


@InputSeries(TextSeries)
def remove_hashtags(s: TextSeries) -> TextSeries:
    """
    Examples
    --------
    >>> import texthero.lang.zh as hero
    >>> import pandas as pd
    >>> s = pd.Series("今天天气真不错#杰克_texthero_123。")
    >>> hero.remove_hashtags(s)
    0    今天天气真不错 。
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.remove_hashtags`
    """
    return replace_hashtags(s, " ")


@InputSeries(TextSeries)
def tokenize(s: TextSeries) -> TokenSeries:
    """
    Examples
    --------
    >>> import texthero.lang.zh as hero
    >>> import pandas as pd
    >>> s = pd.Series(["我昨天吃烤鸭去了。"])
    >>> hero.tokenize(s)
    0    [我, 昨天, 吃, 烤鸭, 去, 了, 。]
    dtype: object

    See also
    --------
    :meth:`texthero.preprocessing.tokenize`
    """
    tokenizer = Chinese()
    return s.apply(lambda string: [token.text for token in tokenizer(string)])
