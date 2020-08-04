<h1 align="center">Pandas Series Types in Texthero</h2>

In Texthero, we're always working with Pandas Series and Pandas Dataframes to hold a (possibly very large) collection of documents. To make things easier and more intuitive, we differentiate between 4 different types of Series, depending on the cell content. For example, the functions in preprocessing.py usually take as input a Series where every cell is a string, and return as output a Series where every cell is a string.

These are the implemented types:

1. TextSeries: Every cell is a text, i.e. a string. For example,
`pd.Series(["test", "test"])` is a valid TextSeries.

2. TokenSeries: Every cell is a list of words/tokens, i.e. a list
of strings. For example, `pd.Series([["test"], ["token2", "token3"]])` is a valid TokenSeries.

3. VectorSeries: Every cell is a vector representing text, i.e.
a list of floats. For example, `pd.Series([[1.0, 2.0], [3.0]])` is a valid VectorSeries.

4. RepresentationSeries: Series is multiindexed with level one
being the document, level two being the individual features and their values.
For example,
`pd.Series([1, 2, 3], index=pd.MultiIndex.from_tuples([("doc1", "word1"), ("doc1", "word2"), ("doc2", "word1")]))`
is a valid RepresentationSeries.

Now, if you see a function in the documentation that looks like this:
```
def tfidf(s: TokenSeries) -> RepresentationSeries
```

then you know that the function takes a Pandas Series
whose cells are lists of strings (tokens) and will
return a Pandas Series whose cells are lists of floats. And this function:
```
def pca(s: Union[VectorSeries, RepresentationSeries) -> VectorSeries
```
can handle both _VectorSeries_ and _RepresentationSeries_ as input and always returns a _VectorSeries_.



<h2 align="center">Representation Series</h2>

As you can see, the `RepresentationSeries` type is a little more complex than the others. Let's have a closer look to see what it is and why, where and how it is used!

<h3 align="left">What is it?</h2>

A _RepresentationSeries_ is multiindexed with level one
being the document, and level two being the individual features and their values. It could look like this:

```python
>>> import texthero as hero
>>> import pandas as pd
>>> s = pd.Series(["Sentence one one", "Sentence two"])
>>> s.pipe(hero.tokenize).pipe(hero.count)  # first tokenize Series, then calculate word count
document  word    
0         Sentence    1
          one         2
1         Sentence    1
          two         1
dtype: Sparse[int64, 0]
```

The output shown is a _RepresentationSeries_! It just means that we have a level for each document, and in that level we can see the individual features of the document.


<h3 align="left">Why is it used?</h2>

You might have noticed the `dtype: Sparse[int64, 0]` in the last code bit. That's precisely the reason we use this: Pandas internally does not store the zeros we get when e.g. calculating `hero.count`; so we don't see that the word "two" has zero occurrences in the first sentence, it's not stored. This is a massive advantage when dealing with *big data*: In a _RepresentationSeries_, we only store the data that's relevant for each document to save time and space!

<h3 align="left">When and how is it used? Do I have to work with multiindexes?!</h2>

The _RepresentationSeries_ is mostly used internally for performance reasons. For example, as you can see above, the default output from `hero.count` is such a Series, but if you apply e.g. `hero.pca` afterwards, you don't even notice the complex _RepresentationSeries_: `s.pipe(hero.count).pipe(hero.pca)` works just fine; everything is seamlessly integrated in the library.

The only thing you cannot do is store a _RepresentationSeries_ in your dataframe, as the indexes are different. If you really want to do this, you can use `hero.flatten`:

```python
>>> import texthero as hero
>>> import pandas as pd
>>> s = pd.Series(["Sentence one one", "Sentence two"])
>>> df = pd.DataFrame(s)
>>> df["count"] = s.pipe(hero.tokenize).pipe(hero.count) # WRONG
>>> # ERROR: cannot put RepresentationSeries into the DataFrame
>>> # INSTEAD DO THIS:
>>> df["count"] = s.pipe(hero.tokenize).pipe(hero.count).pipe(hero.flatten)
>>> df
                  0          count
0  Sentence one one  [1, 2.0, 0.0]
1      Sentence two  [1, 0.0, 1.0]
```
As you can see, we then lose the advantage of _sparseness_ (i.e. not storing the zeroes): The third word, "two", is now also stored for the first sentence with "0.0" occurrences.
