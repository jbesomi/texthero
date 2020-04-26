<a name=".texthero"></a>
## texthero

Texthero: python toolkit for text preprocessing, representation and visualization.

<a name=".texthero.preprocessing"></a>
## texthero.preprocessing

Utility functions to clean text-columns of a dataframe.

<a name=".texthero.preprocessing.fillna"></a>
#### fillna

```python
fillna(input: pd.Series) -> pd.Series
```

Replace not assigned values with empty spaces.

<a name=".texthero.preprocessing.lowercase"></a>
#### lowercase

```python
lowercase(input: pd.Series) -> pd.Series
```

Lowercase all cells.

<a name=".texthero.preprocessing.remove_digits"></a>
#### remove\_digits

```python
remove_digits(input: pd.Series, only_blocks=True) -> pd.Series
```

Remove all digits.

Parameters
----------

input: pd.Series
only_blocks : bool
    Remove only blocks of digits. For instance, `hel1234lo 1234` becomes `hel1234lo`.

Returns
-------

pd.Series

Examples
--------

    >>> import texthero
    >>> s = pd.Series(["remove_digits_s remove all the 1234 digits of a pandas series. H1N1"])
    >>> texthero.preprocessing.remove_digits_s(s)
    u'remove_digits_s remove all the digits of a pandas series. H1N1'
    >>> texthero.preprocessing.remove_digits_s(s, only_blocks=False)
    u'remove_digits_s remove all the digits of a pandas series. HN'

<a name=".texthero.preprocessing.remove_punctuations"></a>
#### remove\_punctuations

```python
remove_punctuations(input: pd.Series) -> pd.Series
```

Remove punctuations from input

<a name=".texthero.preprocessing.remove_diacritics"></a>
#### remove\_diacritics

```python
remove_diacritics(input: pd.Series) -> pd.Series
```

Remove diacritics (as accent marks) from input

<a name=".texthero.preprocessing.remove_spaces"></a>
#### remove\_spaces

```python
remove_spaces(input: pd.Series) -> pd.Series
```

Remove any type of space between words.

<a name=".texthero.representation"></a>
## texthero.representation

Text representation

<a name=".texthero.visualization"></a>
## texthero.visualization

Text visualization

<a name=".texthero.visualization.scatterplot"></a>
#### scatterplot

```python
scatterplot(df: pd.DataFrame, column: str, color: str, hover_data: []) -> None
```

Scatterplot of df[column].

The df[column] must be a tuple of 2d-coordinates.

Usage example:

    >>> import texthero
    >>> df = pd.DataFrame([(0,1), (1,0)], columns='pca')
    >>> texthero.visualization.scatterplot(df, 'pca')

<a name=".texthero.visualization.top_words"></a>
#### top\_words

```python
top_words(s: pd.Series, normalize=True) -> pd.Series
```

Return most common words of a given series sorted from most used.

