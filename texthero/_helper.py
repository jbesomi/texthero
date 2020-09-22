"""
Useful helper functions for the texthero library.
"""

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
    >>> replace_b_with_c(s_with_nan) # doctest: +SKIP
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
