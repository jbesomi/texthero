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


'''
Pandas Integration of DocumentTermDF

It's really important that users can seamlessly integrate texthero's function
output with their code. Let's assume a user has his documents in a DataFrame
`df["texts"]` that looks like this:

```
>>> df = pd.DataFrame(["Text of doc 1", "Text of doc 2", "Text of doc 3"], columns=["text"])
>>> df
            text
0  Text of doc 1
1  Text of doc 2
2  Text of doc 3

```

 Let's look at an example output that `hero.count` could
return with the DocumentTermDF:

```
>>> hero.count(df["text"])
 count                  
      1  2  3 Text doc of
0     1  0  0    1   1  1
1     0  1  0    1   1  1
2     0  0  1    1   1  1
```

That's a DataFrame. Great! Of course, users can
just store this somewhere as e.g. `df_count = hero.count(df["texts"])`,
and that works great. Accessing is then also as always: to get the
count values, they can just do `df_count.values` and have the count matrix
right there!

However, what we see really often is users wanting to do this:
`df["count"] = hero.count(df["texts"])`. This sadly does not work out
of the box. The reason is that this subcolumn type is implemented
internally through a _Multiindex in the columns_. So we have

```
>>> df.columns
Index(['text'], dtype='object')
>>> hero.count(df["texts"]).columns
MultiIndex([('count',    '1'),
            ('count',    '2'),
            ('count',    '3'),
            ('count', 'Text'),
            ('count',  'doc'),
            ('count',   'of')],
           )

```

Pandas _cannot_ automatically combine these. So what we will
do is this: Calling `df["count"] = hero.count(df["texts"])` is
internally this: `pd.DataFrame.__setitem__(self=df, key="count", value=hero.count(df["texts"]))`.
We will overwrite this method so that if _self_ is not multiindexed yet
and _value_ is multiindexed, we transform _self_ (so `df` here) to
be multiindexed and we can then easily integrate our column-multiindexed output from texthero:

If `df` is multiindexed, we get the desired result through `pd.concat([df, hero.count(df["texts"])], axis=1)`.

Pseudocode (& real code): working on this atm :3rd_place_medal: 

Advantages / Why does this work?

    - we don't destroy any pandas functionality as currently calling
      `__setitem__` with a Multiindexed value is just not possible, so
      our changes to Pandas do not break any Pandas functionality for
      the users. We're only _expanding_ the functinoality

    - after multiindexing, users can still access their
      "normal" columns like before; e.g. `df["texts"]` will
      behave the same way as before even though it is now internally
      multiindexed as `MultiIndex([('text', ''), ('count',    '1'),
            ('count',    '2'),
            ('count',    '3'),
            ('count', 'Text'),
            ('count',  'doc'),
            ('count',   'of')],
           )`.

Disadvantage:

    - poor performance, so we discurage user from using it, but we still want to support it
'''

# Store the original __setitem__ function as _original__setitem__
_pd_original__setitem__ = pd.DataFrame.__setitem__
pd.DataFrame._original__setitem__ = _pd_original__setitem__


# Define a new __setitem__ function that will replace pd.DataFrame.__setitem__
def _hero__setitem__(self, key, value):
    '''
    Called when doing self["key"] = value.
    E.g. df["count"] = hero.count(df["texts"]) is internally doing
    pd.DataFrame.__setitem__(self=df, key="count", value=hero.count(df["texts"]).

    So self is df, key is the new column's name, value is
    what we want to put into the new column.

    What we do:

    1. If user calls __setitem__ with value being multiindexed, e.g.
       df["count"] = hero.count(df["texts"]),
       so __setitem__(self=df, key="count", value=hero.count(df["texts"])

        2. we make self multiindexed if it isn't already
            -> e.g. column "text" internally becomes multiindexed
               to ("text", "") but users do _not_ notice this.
               This is a very quick operation that does not need
               to look at the df's values, we just reassign
               self.columns

        3. we change value's columns so the first level is named `key`
            -> e.g. a user might do df["haha"] = hero.count(df["texts"]),
               so just doing df[hero.count(df["texts"]).columns] = hero.count(df["texts"])
               would give him a new column that is named like texthero's output,
               e.g. "count" instead of "haha". So we internally rename the
               value columns (e.g. ('haha',    '1'),
                ('haha',    '2'),
                ('haha',    '3'),
                ('haha', 'Text'),
                ('haha',  'doc'),
                ('haha',   'of')]])

        4. we do self[value.columns] = value as that's exactly the command
           that correctly integrates the multiindexed `value` into `self`

    '''


    # 1.
    if isinstance(value, pd.DataFrame) and isinstance(value.columns, pd.MultiIndex) and isinstance(key, str):

        # 2.
        if not isinstance(self.columns, pd.MultiIndex):
            self.columns = pd.MultiIndex.from_tuples([(col_name, "") for col_name in self.columns.values])

        # 3.
        value.columns = pd.MultiIndex.from_tuples([(key, subcol_name) for _, subcol_name in value.columns.values])

        # 4.
        self[value.columns] = value

    else:

       self._original__setitem__(key, value)


# Replace __setitem__ with our custom function
pd.DataFrame.__setitem__ = _hero__setitem__
