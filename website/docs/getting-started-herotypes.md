<h1 align="center">HeroTypes</h1>

In Texthero, we're always working with Pandas Series and Pandas Dataframes to gain insights from text data! To make things easier and more intuitive, we differentiate between different types of Series/DataFrames, depending on where we are on the road to understanding our dataset.

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

 Consequently, in the Texthero's _preprocessing_ module, the functions usually take as input a Series where every cell is a string, and return as output a Series where every cell is a string. We will call this kind of Series _TextSeries_, so users know immediately what kind of Series the functions can work on. For example, you might see a function
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

4. **DataFrame**: A Pandas DataFrame where the rows can be the documents and the columns can be the words/terms in all the documents.
For example,
`pd.DataFrame([[1, 2, 3], [4,5,6]], columns=["hi", "servus", "hola"])`
is a valid DataFrame.

Now, if you see a function in the documentation that looks like this:
```python
tfidf(s: TokenSeries) -> DataFrame
```

then you know that the function takes a Pandas Series
whose cells are lists of strings (tokens) and will
return a Pandas DataFrame with the individual features.
You might call it like this:
```python
>>> import texthero as hero
>>> import pandas as pd
>>> s = pd.Series(["Text of first document", "Text of second document"])
>>> s_tfidf = s.pipe(hero.tokenize).pipe(hero.tfidf)
```


And this function:
```python
pca(s: Union[VectorSeries, DataFrame]) -> VectorSeries
```
needs a _DataFrame_ or _VectorSeries_ as input and always returns a _VectorSeries_.

<h2 align="center">The Types in Detail</h2>

We'll now have a closer look at each of the types and learn where they are used in a typical NLP workflow.

<h3 align="left">TextSeries</h3>

In a _TextSeries_, every cell is a string. As we saw at the beginning of this tutorial, this type is mostly used in preprocessing. It is very simple and allows us to easily clean a text dataset. Additionally, many NLP functions such as `named_entities, noun_chunks, pos_tag` take a _TextSeries_ as input.

Example of a function that takes and returns a _TextSeries_:
```python
>>> s = pd.Series(["Text: of first! document", "Text of second ... document"])
>>> hero.clean(s)
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

In a _VectorSeries_, every cell is a vector representing text. We use this when we have a low-dimensional (e.g. vectors with length <=1000), dense (so not a lot of zeroes) representation of our texts that we want to work on. For example, the dimensionality reduction functions `pca, nmf, tsne` all take a high-dimensional representation of our text (in the form of a _DataFrame_ (see below) or _VectorSeries_, and return a low-dimensional representation of our text in the form of a _VectorSeries_.

Example of a function that takes as input a _DataFrame_ or _VectorSeries_ and returns a _VectorSeries_:
```python
>>> s = pd.Series(["text first document", "text second document"]).pipe(hero.tokenize).pipe(hero.term_frequency)
>>> hero.pca(s)
0     [0.118, 0.0]
1    [-0.118, 0.0]
dtype: object
```

<h3 align="left">DataFrame</h3>

As you can see, the `DataFrame` type is a little more complex than the others. Let's have a closer look to see what it is and why, where and how it is used!

<h4 align="left">What is it?</h4>

A _DataFrame_ can be a Pandas implementation of a [Document Term Matrix](https://en.wikipedia.org/wiki/Document-term_matrix), so the rows are the documents and the columns are the words/terms in all the documents. Other representations are Topic-Term matrices or Document-Topic matrices. In the Topic modeling tutorial you can find a more detailed introcution into those. 

```python
>>> s = pd.Series(["Sentence one one", "Sentence two"])
>>> t = s.pipe(hero.tokenize).pipe(hero.count)  # first tokenize Series, then calculate word count
>>> t     
  Sentence one two
0        1   2   0
1        1   0   1
```


<h4 align="left">Why is it used?</h4>

Have a look at the datatypes of `t` from the last code bit:
```python
>>> t.dtypes
count  Sentence    Sparse[int64, 0]
       one         Sparse[int64, 0]
       two         Sparse[int64, 0]
``` 
That's precisely the reason we use this: Pandas internally does not store the zeros we get when e.g. calculating `hero.count`; so we don't see that the word "two" has zero occurrences in the first sentence, it's not stored. This is a massive advantage when dealing with *big data*: In a _DataFrame_, we only store the data that's relevant for each document to save lots and lots of time and space!

Let's look at an example with some more data.
```python
>>> data = pd.read_csv("https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv")
>>> data_count = data["text"].pipe(count)
>>> data_count.sparse.density
0.012...
```
We can see that only around 1.2% of our _DataFrame_ `data_count` is filled, so using the sparse DataFrame is saving us a lot of space.

<h4 align="left">When and how is it used? Do I have to work with multiindexes?!</h4>

The _DataFrame_ is mostly used internally for performance reasons. For example, as you can see above, the default output from `hero.count` is such a Pandas DataFrame, but if you apply e.g. `hero.pca` afterwards, you don't even notice the _DataFrame_: `s.pipe(hero.count).pipe(hero.normalize).pipe(hero.pca)` works just fine; everything is seamlessly integrated in the library.

The only thing you _can_ but _should not_ do is store a _DataFrame_ in your dataframe, as the performance is really bad. If you really want to, you can do it like this like this:
```python
>>> data = pd.read_csv("https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv")
>>> data_count = data["text"].pipe(count)

>>> # NOT recommended as performance is not optimal
>>> data["count"] = data_count
```