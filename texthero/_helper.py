"""
Useful helper functions for the texthero library.
"""

import wrapt
import pandas as pd
import functools
import warnings


"""
Warnings.
"""

_warning_nans_in_input = (
    "There are NaNs (missing values) in the given input series."
    " They were replaced with appropriate values before the function"
    " was applied. Consider using hero.fillna to replace those NaNs yourself"
    " or hero.drop_no_content to remove them."
)


"""
Decorators.
"""


def handle_nans(replace_nans_with):
    """
    Decorator to handle NaN values in a function's input.

    Using the decorator, if there are NaNs in the input,
    they are replaced with replace_nans_with
    and a warning is printed.

    The function must take as first input a Pandas Series.

    Examples
    --------
    >>> from texthero._helper import handle_nans
    >>> import pandas as pd
    >>> import numpy as np
    >>> @handle_nans(replace_nans_with="I was missing!")
    ... def replace_b_with_c(s):
    ...     return s.str.replace("b", "c")
    >>> s_with_nan = pd.Series(["Test b", np.nan])
    >>> replace_b_with_c(s_with_nan)
    0            Test c
    1    I was missing!
    dtype: object
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            # Get first input argument (the series) and replace the NaNs.
            s = args[0]
            if s.isna().values.any():
                warnings.warn(_warning_nans_in_input, UserWarning)
                s = s.fillna(value=replace_nans_with)

            # Put the series back into the input.
            if args[1:]:
                args = (s,) + args[1:]
            else:
                args = (s,)

            # Apply function as usual.
            return func(*args, **kwargs)

        return wrapper

    return decorator


"""
Hero Series Types
=================

There are different kinds of Pandas Series used in the library, depending on use.
For example, the functions in preprocessing.py usually take as input a Series
where every cell is a string, and return as output a Series where every cell
is a string. To make handling the different types easier (and most importantly
intuitive for users), this file implements the types as subclasses of Pandas
Series and defines functions to check the types.

The goal is to be able to do something like this:

@OutputSeries(DocumentRepresentationSeries)
@InputSeries(TokenSeries)
def tfidf(s: TokenSeries) -> DocumentRepresentationSeries:
    ...

The decorator (@...) makes python check whether the input is
of correct form/type,
and whether the output is of correct form/type,
which leads to easier code and exception handling (no need to write
"if not is_text_series(s): raise ..." in every function) and easy
modification/expansion later on.

The typing helps the users understand the code more easily
as they'll be able to see immediately from the documentation
on what types of Series a function operates. This is much more
verbose and clearer than e.g. "tfidf(s: pd.Series) -> pd.Series".

Note that users can and should of course still simply 
use ordinary pd.Series objects. The custom types are just subclasses of pd.Series so
we can easily use them in the code.

This is mainly for
- ease of development: just put the correct decorators and
don't think about printing errors for incorrect types
- ease of use / understanding: better typing/documentation for users


These are the implemented types:

- TextSeries: cells are text (i.e. strings), e.g. "Test"
- TokenSeries: cells are lists of tokens (i.e. lists of strings), e.g. ["word1", "word2"]
- VectorSeries: cells are vector representations of text, e.g. [0.25, 0.75]
- DocumentRepresentationSeries: Series is multiindexed with level one
being the document, level two being the individual features and their values

The classes are lightweight subclasses of pd.Series and serve 2 purposes:
1. Good documentation for users through docstring.
2. Function to check if a pd.Series has the required properties.

"""


"""
The Hero Series classes.
"""

# This class is mainly for documentation in the docstring.


class HeroSeries(pd.Series):
    """
    Hero Series Types
    =================
    In texthero, most functions operate on a Pandas Series as input
    and give a Pandas Series as output. There are currently four
    main types of Series' in use, which are supported as classes
    by the library:

    1. TextSeries: Every cell is a text, i.e. a string. For example,
    `pd.Series(["test", "test"])` is a valid TextSeries.

    2. TokenSeries: Every cell is a list of words/tokens, i.e. a list
    of strings. For example, `pd.Series([["test"], ["token2", "token3"]])` is a valid TokenSeries.

    3. VectorSeries: Every cell is a vector representing text, i.e.
    a list of floats. For example, `pd.Series([[1.0, 2.0], [3.0]])` is a valid VectorSeries.

    4. DocumentRepresentationSeries: Series is multiindexed with level one
    being the document, level two being the individual features and their values.
    For example,
    `pd.Series([1, 2, 3], index=pd.MultiIndex.from_tuples([("doc1", "word1"), ("doc1", "word2"), ("doc2", "word1")]))`
    is a valid DocumentRepresentationSeries.

    These types of Series are supposed to make using the library
    easier and more intuitive. For example, if you see a
    function head
    ```
    def tfidf(s: TokenSeries) -> DocumentRepresentationSeries
    ```
    then you know that the function takes a Pandas Series
    whose cells are lists of strings (tokens) and will
    return a Pandas Series whose cells are vectors of floats.
    """

    @staticmethod
    def check_series():
        raise NotImplementedError()  # Every Hero Series type has to have this.


class TextSeries(HeroSeries):
    """
    In a TextSeries, every cell has to be a text, i.e. a string. For example,
    `pd.Series(["test", "test"])` is a valid TextSeries.
    """

    @staticmethod
    def check_series(s: pd.Series) -> bool:
        """
        Check if a given Pandas Series has the properties of a TextSeries.
        """

        error_string = (
            "The input Series should consist only of strings in every cell."
            " See help(hero.HeroSeries) for more information."
        )

        if not isinstance(s.iloc[0], str) or s.index.nlevels != 1:
            raise TypeError(error_string)


class TokenSeries(HeroSeries):
    """
    In a TokenSeries, every cell has to be a list of words/tokens, i.e. a list
    of strings. For example, `pd.Series([["test"], ["token2", "token3"]])` is a valid TokenSeries.
    """

    @staticmethod
    def check_series(s: pd.Series) -> bool:
        """
        Check if a given Pandas Series has the properties of a TokenSeries.
        """

        error_string = (
            "There are non-token cells (every cell should be a list of words/tokens) in the given Series."
            " See help(hero.HeroSeries) for more information."
        )

        def is_list_of_strings(cell):
            return all(isinstance(x, str) for x in cell) and isinstance(
                cell, (list, tuple)
            )

        if not is_list_of_strings(s.iloc[0]) or s.index.nlevels != 1:
            raise TypeError(error_string)


class VectorSeries(HeroSeries):
    """
    In a VectorSeries, every cell is a vector representing text, i.e.
    a list of numbers.
    For example, `pd.Series([[1.0, 2.0], [3.0]])` is a valid VectorSeries.
    """

    @staticmethod
    def check_series(s: pd.Series, input_output="") -> bool:
        """
        Check if a given Pandas Series has the properties of a RepresentationSeries.
        """

        error_string = (
            "There are non-representation cells (every cell should be a list of floats) in the given Series."
            " See help(hero.HeroSeries) for more information."
        )

        def is_numeric(x):
            try:
                float(x)
            except ValueError:
                return False
            else:
                return True

        def is_list_of_numbers(cell):
            return all(is_numeric(x) for x in cell) and isinstance(cell, (list, tuple))

        if not is_list_of_numbers(s.iloc[0]) or s.index.nlevels != 1:
            raise TypeError(error_string)


class DocumentRepresentationSeries(HeroSeries):
    """
    A DocumentRepresentationSeries is multiindexed with level one
    being the document, and level two being the individual features and their values.
    For example,
    `pd.Series([1, 2, 3], index=pd.MultiIndex.from_tuples([("doc1", "word1"), ("doc1", "word2"), ("doc2", "word1")]))`
    is a valid DocumentRepresentationSeries.
    """

    @staticmethod
    def check_series(s: pd.Series, input_output="") -> bool:
        """
        Check if a given Pandas Series has the properties of a DocumentRepresentationSeries.
        """

        error_string = (
            "The input Pandas Series should be a Representation Pandas Series and should have a MultiIndex."
            " See help(hero.HeroSeries) for more information."
        )

        if not isinstance(s.index, pd.MultiIndex) or s.index.nlevels != 2:
            raise TypeError(error_string)


def InputSeries(allowed_hero_series_type):
    """
    Check if first argument of function has / fulfills
    type allowed_hero_series_type

    Examples
    --------
    >>> from texthero._helper import *
    >>> import pandas as pd
    >>> @InputSeries(TokenSeries)
    ... def f(s):
    ...     pass
    >>> f(pd.Series("Not tokenized")) # doctest: +SKIP
    >>> # throws a type error with a nice explaination
    >>> f(pd.Series([["I", "am", "tokenized"]]))
    >>> # passes
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            s = args[0]  # The first input argument will be checked.
            # Check if input series can fulfill type.
            allowed_hero_series_type.check_series(s)

            # If we get here, the type can be fulfilled -> execute function as usual.
            return func(*args, **kwargs)

        return wrapper

    return decorator
