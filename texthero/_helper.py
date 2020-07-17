"""
Useful helper functions for the texthero library.
"""

import functools
import wrapt
import numpy as np


"""
Decorators.
"""


def handle_nans(wrapped=None, input_only=False):
    """
    Decorator to make a function not change NaN values.

    Using the decorator, the function to be applied
    will not change cells that have value np.nan.

    The function must take as first input a Series s,
    manipulate that Series (e.g. removing diacritics)
    and then return as first output the Series s.

    Parameters
    ----------
    input_only: Boolean, default to False.
        Set to True when the output that is returned by the
        function is _not_ the same as the input series
        with (some) cells changed (e.g. in top_words,
        the output Series is different from the input
        Series, and in pca there is no return, so in both
        cases input_only is set to True).


    Examples
    --------
    >>> from texthero._helper import *
    >>> import pandas as pd
    >>> import numpy as np
    >>> @handle_nans
    ... def replace_a_with_b(s):
    ...     return s.str.replace("a", "b")
    >>> s_with_nan = pd.Series(["Test a", np.nan])
    >>> replace_a_with_b(s_with_nan)
    0    Test b
    1       NaN
    dtype: object
    """
    if wrapped is None:
        return functools.partial(handle_nans, input_only=input_only)

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):

        # Get first input argument (the series).
        s = args[0]
        nan_mask = ~s.isna()

        # Need a copy as changing s[nan_mask] would change the original input.
        s_result = s.copy()
        s_without_nans = s[nan_mask]

        # Change input Series so the function will only work on the non-nan fields.
        args = (s_without_nans,) + args[1:] if args[1:] else (s_without_nans,)

        # Execute the function and get the result.
        output = wrapped(*args, **kwargs)

        # If we should also handle the output.
        if not input_only:
            # Replace first argument of output (that's the Series) to refill the NaN fields.
            if not isinstance(output, tuple):
                output = (output,)
            s_result[nan_mask] = output[0]

            # Recover index name if set.
            if output[0].index.name:
                s_result.index.name = output[0].index.name

            output = (s_result,) + output[1:] if output[1:] else s_result

        return output

    return wrapper(wrapped)
