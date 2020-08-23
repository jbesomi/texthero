"""
Useful helper functions for the texthero library.
"""
import sys
import pandas as pd
import multiprocessing as mp
import numpy as np
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





def parallelize2(func):
    """
    TODO
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        # Get first input argument (the series).
        s = args[0]

        # If enough rows for us to parallelize
        if len(s) > MIN_LINES_FOR_PARALLELIZATION:

            partitions = mp.cpu_count()

            # split data into batches
            data_split = np.array_split(s, partitions)

            # open threadpool
            pool = mp.Pool(partitions)

            # Execute in parallel and concat results (order is kept).
            s_result = pd.concat(
                pool.map(functools.partial(func, *args, **kwargs), data_split)
            )

            pool.close()
            pool.join()

            return s_result

        else:
            # Apply function as usual.
            return func(*args, **kwargs)

    setattr(sys.modules[func.__module__], func.__name__, wrapper)

    return wrapper


#def _f(s, t):
#    return s.str.split()



#def f(*args, **kwargs)= parallelize(_f, *args, **kwargs)


MIN_LINES_FOR_PARALLELIZATION = 0


def doit(s, f):
    return s.apply(f)

import re

#def g(s):
#    return s.apply(lambda x: re.sub(r"j", "br", x))

"""
lambda x: _map_apply
"""

def _hero_apply(s, func, *args, **kwargs):

    # If enough rows for us to parallelize
    if len(s) > MIN_LINES_FOR_PARALLELIZATION:

        partitions = mp.cpu_count()

        # split data into batches
        data_split = np.array_split(s, partitions)

        # open threadpool
        pool = mp.Pool(partitions)

        # Execute in parallel and concat results (order is kept).
        s_result = pd.concat(
            pool.map(func, data_split)
        )

        pool.close()
        pool.join()

        return s_result

    else:
        # Apply function as usual.
        return func(s, *args, **kwargs)



"""
s = _hero_apply(s, lambda x: re.sub(r"", "", x))

def _hero_apply:
    split up s
    for s_partial in s:
        s_partial_result = s_partial.apply()
"""


import numpy as np
from multiprocessing import cpu_count
 
cores = cpu_count() #Number of CPU cores on your system
partitions = cores #Define as many partitions as you want
 
def parallel(data, func, *args, **kwargs):

    data_split = np.array_split(data, partitions)
    pool = mp.Pool(cores)
    data = pd.concat(
                pool.map(
                    functools.partial(func, *args, **kwargs), data_split
                )
            )

    pool.close()
    pool.join()
    return data

"""
import texthero as t
from texthero._helper import g
import pandas as pd
s = pd.Series(["Ja1a 9" for _ in range(10)])

t._helper.parallel(s, t.preprocessing.replace_digits, symbols="nee", only_blocks=False)
"""