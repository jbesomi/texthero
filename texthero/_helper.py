"""
Useful internal helper functions for the library.
"""

import pandas as pd
import wrapt

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

@OutputSeries(RepresentationSeries)
@InputSeries(TokenSeries)
def tfidf(s: TokenSeries) -> RepresentationSeries:
    ...

The decorators (@...) make python check whether the input is valid
and transform the output into the correct type,
which leads to easier code and exception handling (no need to write
"if not is_text_series(s): raise ..." in every function) and easy
modification/expansion later on.

The typing helps the users understand the code more easily
as they'll be able to see immediately from the documentation
on what types of Series a function operates. This is much more
verbose and clearer than e.g. "tfidf(s: pd.Series) -> pd.Series".

Note that users can of course still simply 
use ordinary pd.Series objects.
The functions will then just check if the Series _could be_
e.g. a TextSeries (so it checks the properties) to give maximum flexibility.
The custom types are subclasses of pd.Series anyway. Thus,
the types enable better documentation and expressiveness
of the code and _do not_ mean that a user really has to pass
a e.g. TextSeries; what he passes just has to have the properties
of one.

Example: user has standard pd.Series s and wants to clean the text.
Calling hero.clean(s), the clean function will check whether s
_could be_ a TextSeries. If yes, it proceeds with the cleaning
and returns a TextSeries. If no, an error is thrown with
a good explaination.

Concerning performance, a user might often have a Series s on which
different operations will be performed. The behaviour will be as follows:
>> s = pd.Series("test")
>> s = hero.remove_punctuation(s)
>> # hero.remove_punctuation first checked if s can be a TextSeries.
>> # That is the case, so the function was applied as usual.
>> # The output was then transformed to a TextSeries, without
>> # the user noticing. If now something like this is done:
>> s = hero.remove_diacritics(s)
>> # the remove_diacritics function will immediately notice
>> # that s is a TextSeries, so the check is O(1) through isinstance.

(NOTE: this could lead to problems later on, if e.g. a user
changes s after remove_punctuation, then the library still
treats it as a TextSeries even though the user might have
applied functions from e.g a different library such that s does not
fulfill the "TextSeries" properties anymore. The error messages
would then be not as good.)

These are the implemented types:

- TextSeries: cells are text (i.e. strings), e.g. "Test"
- TokenSeries: cells are lists of tokens (i.e. lists of strings), e.g. ["word1", "word2"]
- RepresentationSeries: cells are vector representations of text (see issue #43), e.g. [0.25, 0.75]

The classes are lightweight subclasses of pd.Series and serve 2 purposes:
1. Good documentation for users through docstring.
2. Function(s) to check if a pd.Series has the required properties.

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
    and give a Pandas Series as output. There are currently three
    main types of Series' in use, which are supported as classes
    by the library:

    1. TextSeries: Every cell is a text, i.e. a string. For example,
                   pd.Series(["test", "test"]) is a valid TextSeries.
    2. TokenSeries: Every cell is a list of words/tokens, i.e. a list
                    of strings. For example, 
                    pd.Series([["test"], ["token2", "token3"]]) is a valid TokenSeries.
    3. RepresentationSeries: Every cell is a vector representing text, i.e.
                             a list of floats. For example,
                             pd.Series([[1.0, 2.0], [3.0]]) is a valid RepresentationSeries.

    These types of Series are supposed to make using the library
    easier and more intuitive. For example, if you see a
    function head
    ```
    def tfidf(s: TokenSeries) -> RepresentationSeries
    ```
    then you know that the function takes a Pandas Series
    whose cells are lists of strings (tokens) and will
    return a Pandas Series whose cells are vectors of floats.
    """

    @staticmethod
    def check_series():
        raise NotImplementedError()  # Every Hero Series type has to have this.


def _convert_series(s, target_series_type):
    """ Function to convert a series to a target type if it isn't already."""
    if isinstance(s, target_series_type):
        return s
    else:
        return target_series_type(s)


class TextSeries(HeroSeries):
    """
    In a TextSeries, every cell has to be a text, i.e. a string. For example,
    pd.Series(["test", "test"]) is a valid TextSeries.
    """

    @staticmethod
    def check_series(s: pd.Series, input_output="") -> bool:
        """
        Check if a given Pandas Series has the properties of a TextSeries.
        """

        error_string = (
            "There are non-string cells (every cell should be a string) in the given Series."
            " See help(hero.HeroSeries) for more information."
            " You might want to use hero.drop_no_contents(s) to drop missing values."
        ).format(input_output)

        if isinstance(s, TextSeries):
            return

        if not isinstance(s, pd.Series):
            raise TypeError("{} type must be Pandas Series.".format(input_output))

        try:
            if not s.map(type).eq(str).all():
                raise TypeError(error_string)
        except:
            raise TypeError(error_string)


class TokenSeries(HeroSeries):
    """
    In a TokenSeries, every cell has to be a list of words/tokens, i.e. a list
    of strings. For example, pd.Series([["test"], ["token2", "token3"]]) is a valid TokenSeries.
    """

    @staticmethod
    def check_series(s: pd.Series, input_output="") -> bool:
        """
        Check if a given Pandas Series has the properties of a TokenSeries.
        """

        error_string = (
            "There are non-token cells (every cell should be a list of words/tokens) in the given {} Series."
            " See help(hero.HeroSeries) for more information."
            " You might want to use hero.tokenize(s) first to tokenize your Series."
        ).format(input_output)

        if isinstance(s, TokenSeries):
            return

        if not isinstance(s, pd.Series):
            raise TypeError("{} type must be Pandas Series.".format(input_output))

        def is_list_of_strings(cell):
            return all(isinstance(s, str) for s in cell) and isinstance(cell, (list, tuple))

        try:
            if not s.map(lambda cell: is_list_of_strings(cell)).all():
                raise TypeError(error_string)
        except:
            raise TypeError(error_string)


class RepresentationSeries(HeroSeries):
    """
    In a RepresentationSeries, every cell is a vector representing text, i.e.
    a list of floats. For example, pd.Series([[1.0, 2.0], [3.0]]) is a valid RepresentationSeries.
    """

    @staticmethod
    def check_series(s: pd.Series, input_output="") -> bool:
        """
        Check if a given Pandas Series has the properties of a RepresentationSeries.
        """

        error_string = (
            "There are non-representation cells (every cell should be a list of floats) in the given {} Series."
            " See help(hero.HeroSeries) for more information."
            " You might want to use a function from hero.representation first to get a representation of your Series."
        ).format(input_output)

        if isinstance(s, RepresentationSeries):  # If already the correct type.
            return

        if not isinstance(s, pd.Series):
            raise TypeError("{} type must be Pandas Series.".format(input_output))

        try:
            if not s.map(lambda cell: all(isinstance(s, float) for s in cell)).all():
                raise TypeError(error_string)
        except:
            raise TypeError(error_string)


"""
The Hero Series decorators.

If a function takes several arguments, the
decorator InputSeries will check the first one.
If there are several outputs, the decorator
OutputSeries will check the first one.
"""


def InputSeries(allowed_hero_series_type):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        print(wrapped.__name__)
        print("inputargs", args)
        s = args[0]  # The first input argument will be checked.
        # Check if input series can fulfill type.
        allowed_hero_series_type.check_series(s, "input")
        # If we get here, the type can be fulfilled -> execute function as usual.
        return wrapped(*args, **kwargs)

    return wrapper


def OutputSeries(allowed_hero_series_type):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        # First execute the function together with the InputSeries decorator.
        output = wrapped(*args, **kwargs)
        if not isinstance(output, tuple):
            output = (output,)

        # Now check first argument of output.
        s = output[0]
        # Check if output series can fulfill type.
        allowed_hero_series_type.check_series(s, "output")
        # If we get here, the type can be fulfilled -> convert to the type (if not already of the type).
        s = _convert_series(s, allowed_hero_series_type)

        # Combine if more than one output.
        return (s,) + output[1:] if output[1:] else s

    return wrapper
