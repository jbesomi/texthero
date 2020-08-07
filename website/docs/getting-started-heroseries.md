<h1 align="center">HeroSeries</h1>

In Texthero, we're always working with Pandas Series and Pandas Dataframes to gain insights from text data! To make things easier and more intuitive, we differentiate between different types of Series, depending on where we are on the road to understanding our dataset.

<h2 align="center">Overview</h2>

When working with text data, it is easy to get overwhelmed by the many different functions that can be applied to the data. We want to make the whole journey as clear as possible. For example, when we start working with a new dataset, we usually want to do some preprocessing first. At the beginning, the data is in a DataFrame or Series where every document is one string. It might look like this:
```python
                                    text
document_id                             
0            "Text in the first document"
1            "Text in the second document"
2            "Text in the third document"
3            "Text in the fourth document"
4                                    ...

```

 Consequently, in the Texthero's _preprocessing_ module, the functions usually take as input a Series where every cell is a string, and return as output a Series where every cell is a string. We will call these kinds of Series _TextSeries_, so users know immediately what kind of Series the functions can work on. For example, you might see a function
 ```python
remove_punctuation(s: TextSeries) -> TextSeries
 ```
in the documentation. You then know that this can be used on a DataFrame or Series in the preprocessing phase of your work, where each document is one string.

<h3 align="center">The four HeroSeries Types</h3>

These are the four types currently supported by the library; almost all of the libraries functions takes as input and return as output one of these types:

1. **TextSeries**: Every cell is a text, i.e. a string. For example,
`pd.Series(["test", "test"])` is a valid TextSeries.

2. **TokenSeries**: Every cell is a list of words/tokens, i.e. a list
of strings. For example, `pd.Series([["test"], ["token2", "token3"]])` is a valid TokenSeries.

3. **VectorSeries**: Every cell is a vector representing text, i.e.
a list of floats. For example, `pd.Series([[1.0, 2.0], [3.0]])` is a valid VectorSeries.

4. **RepresentationSeries**: Series is multiindexed with level one
being the document, level two being the individual features and their values.
For example,
`pd.Series([1, 2, 3], index=pd.MultiIndex.from_tuples([("doc1", "word1"), ("doc1", "word2"), ("doc2", "word1")]))`
is a valid RepresentationSeries.

Now, if you see a function in the documentation that looks like this:
```python
tfidf(s: TokenSeries) -> RepresentationSeries
```

then you know that the function takes a Pandas Series
whose cells are lists of strings (tokens) and will
return a Pandas Series whose cells are lists of floats.
You might call it like this:
```python
>>> import texthero as hero
>>> import pandas as pd
>>> s = pd.Series(["Text of first document", "Text of second document"])
>>> s_tfidf = s.pipe(hero.tokenize).pipe(hero.tfidf)
```


And this function:
```python
pca(s: RepresentationSeries) -> VectorSeries
```
needs a _RepresentationSeries_ as input and always returns a _VectorSeries_.

<h2 align="center">The Types in Detail</h2>

We'll now have a closer look at each of the types and learn where they are used in a typical NLP workflow.

<h3 align="left">TextSeries</h3>

In a _TextSeries_, every cell is a string. As we saw at the beginning of this tutorial, this type is mostly used in preprocessing. It is very simple and allows us to easily clean a text dataset. Additionally, many NLP functions such as `named_entities, noun_chunks, pos_tag` take a _TextSeries_ as input.

Example of a function that takes and returns a _TextSeries_:
```python
>>> s = pd.Series(["Text: of first! document", "Text of second ... document"])
>>> hero.remove_punctuation(s)
0     text first document
1    text second document
dtype: object
```

<h3 align="left">TokenSeries</h3>

In a _TokenSeries_, every cell is a list of words/tokens. We use this to prepare our data for _representation_, so to gain insights from it through mathematical methods. This is why the functions that initially transform your documents to vectors, namely `tfidf, term_frequency, count`, take a _TokenSeries_ as input.

Example of a function that takes a _TextSeries_ and returns a _TokenSeries_:
```python
>>> s = pd.Series(["text first document", "text second document"])
>>> hero.tokenize(s)
0     [text, first, document]
1    [text, second, document]
dtype: object
```

<h3 align="left">VectorSeries</h3>

In a _VectorSeries_, every cell is a vector representing text. We use this when we have a low-dimensional (e.g. lists with length 2 or 3), dense (so not a lot of zeroes) representation of our texts that we want to work on. For example, the dimensionality reduction functions `pca, nmf, tsne` all take a high-dimensional representation of our text in the form of a _RepresentationSeries_ (see below), and return a low-dimensional representation of our text in the form of a _VectorSeries_.

Example of a function that takes as input a _RepresentationSeries_ and returns a _VectorSeries_:
```python
>>> s = pd.Series(["text first document", "text second document"]).pipe(hero.tokenize).pipe(hero.term_frequency)
>>> hero.pca(s)
0     [0.118, 0.0]
1    [-0.118, 0.0]
dtype: object
```

<h3 align="left">Representation Series</h3>

As you can see, the `RepresentationSeries` type is a little more complex than the others. Let's have a closer look to see what it is and why, where and how it is used!

<h4 align="left">What is it?</h4>

A _RepresentationSeries_ is multiindexed with level one
being the document, and level two being the individual features and their values. It could look like this:

```python
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


<h4 align="left">Why is it used?</h4>

You might have noticed the `dtype: Sparse[int64, 0]` in the last code bit. That's precisely the reason we use this: Pandas internally does not store the zeros we get when e.g. calculating `hero.count`; so we don't see that the word "two" has zero occurrences in the first sentence, it's not stored. This is a massive advantage when dealing with *big data*: In a _RepresentationSeries_, we only store the data that's relevant for each document to save time and space!

<h4 align="left">When and how is it used? Do I have to work with multiindexes?!</h4>

The _RepresentationSeries_ is mostly used internally for performance reasons. For example, as you can see above, the default output from `hero.count` is such a Series, but if you apply e.g. `hero.pca` afterwards, you don't even notice the complex _RepresentationSeries_: `s.pipe(hero.count).pipe(hero.pca)` works just fine; everything is seamlessly integrated in the library.

The only thing you cannot do is store a _RepresentationSeries_ in your dataframe, as the indexes are different. If you really want to do this, you can use `hero.flatten`:

```python
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
As you can see, we then lose the advantage of _sparseness_ (i.e. not storing the zeroes): The third word, "two", is now also stored for the first sentence with "0.0" occurrences. This is why we strongly suggest you avoid using `hero.flatten` at all; Texthero is designed with performance in mind and using `hero.flatten` goes against that goal!
