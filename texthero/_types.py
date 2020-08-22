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

@InputSeries(TokenSeries)
def tfidf(s: TokenSeries) -> DocumentTermDF:
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
verbose and clearer than e.g. "tfidf(s: pd.Series) -> pd.DataFrame".

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
- DocumentTermDF: DataFrame is sparse and multiindexed in the columns with every subcolumn
                  being an individual feature

The classes are lightweight subclasses of pd.Series and serve 2 purposes:
1. Good documentation for users through docstring.
2. Function to check if a pd.Series has the required properties.

"""

import functools
import pandas as pd

from typing import Tuple


"""
The Hero Series classes.
"""

# This class is mainly for documentation in the docstring.


class HeroTypes(pd.Series, pd.DataFrame):
    """
    Hero Series Types
    =================
    In texthero, most functions operate on a Pandas Series as input
    and give a Pandas Series as output. There are currently four
    main types of Series / DataFrames in use, which are supported as classes
    by the library:

    1. TextSeries: Every cell is a text, i.e. a string. For example,
    `pd.Series(["test", "test"])` is a valid TextSeries.

    2. TokenSeries: Every cell is a list of words/tokens, i.e. a list
    of strings. For example, `pd.Series([["test"], ["token2", "token3"]])` is a valid TokenSeries.

    3. VectorSeries: Every cell is a vector representing text, i.e.
    a list of floats. For example, `pd.Series([[1.0, 2.0], [3.0]])` is a valid VectorSeries.

    4. DocumentTermDF: DataFrame is sparse and multiindexed in the columns with every subcolumn
    being an individual feature
    For example,
    `pd.DataFrame([[1, 2, 3], [4,5,6]], columns=pd.MultiIndex.from_tuples([("count", "hi"), ("count", "servus"), ("count", "hola")]))`
    is a valid DocumentTermDF.

    These types of Series are supposed to make using the library
    easier and more intuitive. For example, if you see a
    function head
    ```
    def tfidf(s: TokenSeries) -> DocumentTermDF
    ```
    then you know that the function takes a Pandas Series
    whose cells are lists of strings (tokens) and will
    return a sparse Pandas DataFrame where every subcolumn is one feature
    (in this case one word).
    """

    @staticmethod
    def check_type():
        raise NotImplementedError()  # Every Hero Series type has to have this.


class TextSeries(HeroTypes):
    """
    In a TextSeries, every cell has to be a text, i.e. a string. For example,
    `pd.Series(["test", "test"])` is a valid TextSeries.
    """

    @staticmethod
    def check_type(s: pd.Series) -> Tuple[bool, str]:
        """
        Check if a given Pandas Series has the properties of a TextSeries.
        """

        error_string = (
            "should be TextSeries: the input Series should consist only of strings in every cell."
            " See help(hero.HeroTypes) for more information."
        )

        if not (isinstance(s, pd.Series) and isinstance(s.iloc[0], str)):
            return False, error_string
        else:
            return True, ""


class TokenSeries(HeroTypes):
    """
    In a TokenSeries, every cell has to be a list of words/tokens, i.e. a list
    of strings. For example, `pd.Series([["test"], ["token2", "token3"]])` is a valid TokenSeries.
    """

    @staticmethod
    def check_type(s: pd.Series) -> Tuple[bool, str]:
        """
        Check if a given Pandas Series has the properties of a TokenSeries.
        """

        error_string = (
            "should be TokenSeries: there are non-token cells (every cell should be a list of words/tokens) in the given Series."
            " See help(hero.HeroTypes) for more information."
        )

        def is_list_of_strings(cell):
            return all(isinstance(x, str) for x in cell) and isinstance(
                cell, (list, tuple)
            )

        if not (isinstance(s, pd.Series) and is_list_of_strings(s.iloc[0])):
            return False, error_string
        else:
            return True, ""


class VectorSeries(HeroTypes):
    """
    In a VectorSeries, every cell is a vector representing text, i.e.
    a list of numbers.
    For example, `pd.Series([[1.0, 2.0], [3.0]])` is a valid VectorSeries.
    """

    @staticmethod
    def check_type(s: pd.Series, input_output="") -> Tuple[bool, str]:
        """
        Check if a given Pandas Series has the properties of a VectorSeries.
        """

        error_string = (
            "should be VectorSeries: there are non-representation cells (every cell should be a list of floats) in the given Series."
            " See help(hero.HeroTypes) for more information."
        )

        def is_numeric(x):
            try:
                float(x)
            except ValueError:
                return False
            else:
                return True

        def is_list_of_numbers(cell):
            return isinstance(cell, (list, tuple)) and all(is_numeric(x) for x in cell)

        if not (isinstance(s, pd.Series) and is_list_of_numbers(s.iloc[0])):
            return False, error_string
        else:
            return True, ""


class DocumentTermDF(HeroTypes):
    """
    A DocumentTermDF is a sparse DataFrame that is
    multiindexed in the columns with every subcolumn
    being an individual feature.
    For example,
    `pd.DataFrame([[1, 2, 3], [4,5,6]], columns=pd.MultiIndex.from_tuples([("count", "hi"), ("count", "servus"), ("count", "hola")]))`
    is a valid DocumentTermDF.
    """

    @staticmethod
    def check_type(df: pd.DataFrame, input_output="") -> Tuple[bool, str]:
        """
        Check if a given Pandas Series has the properties of a DocumentTermDF.
        """

        error_string = (
            "should be DocumentTermDF: The input should be a DocumentTermDF, so a DataFrame"
            " with a MultiIndex in the columns."
            " See help(hero.HeroTypes) for more information."
        )

        if not (
            isinstance(df, pd.DataFrame)
            and isinstance(df.columns, pd.MultiIndex)
            and pd.api.types.is_sparse(df.dtypes[0])
        ):
            return False, error_string
        else:
            return True, ""


def InputSeries(allowed_hero_series_types):
    """
    Check if first argument of function has / fulfills
    type allowed_hero_series_type

    Examples
    --------
    >>> from texthero._types import *
    >>> import pandas as pd
    >>> @InputSeries(TokenSeries)
    ... def f(s):
    ...     pass
    >>> f(pd.Series("Not tokenized")) # doctest: +SKIP
    >>> # throws a type error with a nice explaination
    >>> f(pd.Series([["I", "am", "tokenized"]]))
    >>> # passes

    With several possible types:

    >>> @InputSeries([DocumentTermDF, VectorSeries])
    ... def g(x):
    ...     pass
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            s = args[0]  # The first input argument will be checked.
            # Check if input series can fulfill type.

            # list -> several possible types
            if isinstance(allowed_hero_series_types, list):

                # Output of check_type is always Bool, Error_String where the Bool is True
                # if the type is fulfilled, else false.
                # if no type is fulfilled (so check_type first output is False for all allowed types),
                # combine all the error strings to show the user all allowed types in the TypeError.
                if not any(
                    allowed_type.check_type(s)[0]
                    for allowed_type in allowed_hero_series_types
                ):

                    error_string = (
                        "Possible types:\n\nEither "
                        + allowed_hero_series_types[0].check_type(s)[1]
                    )

                    for allowed_type in allowed_hero_series_types[1:]:
                        error_string += "\n\nOr " + allowed_type.check_type(s)[1]

                    raise TypeError(error_string)

            else:  # only one possible type
                fulfills, error_string = allowed_hero_series_types.check_type(s)
                if not fulfills:
                    raise TypeError(error_string)

            # If we get here, the type can be fulfilled -> execute function as usual.
            return func(*args, **kwargs)

        return wrapper

    return decorator
