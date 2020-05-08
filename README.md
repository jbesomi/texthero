<p align="center">
   <a href="https://github.com/jbesomi/texthero/stargazers">
    <img src="https://img.shields.io/github/stars/jbesomi/texthero.svg?colorA=orange&colorB=orange&logo=github"
         alt="Github stars">
   </a>
   <a href="https://pypi.org/search/?q=texthero">
      <img src="https://img.shields.io/pypi/v/texthero.svg?colorB=brightgreen"
           alt="pip package">
   </a>
   <a href="https://pypi.org/project/texthero/">
      <img alt="pip downloads" src="https://img.shields.io/pypi/dm/texthero">
   </a>
   <a href="https://github.com/jbesomi/texthero/issues">
        <img src="https://img.shields.io/github/issues/jbesomi/texthero.svg"
             alt="Github issues">
   </a>
   <a href="https://github.com/jbesomi/texthero/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/jbesomi/texthero.svg"
             alt="Github license">
   </a>   
</p>

<p align="center">
    <img src="https://github.com/jbesomi/texthero/raw/master/github/logo.png">
</p>

<p style="font-size: 20px;" align="center">Text preprocessing, representation and visualization from zero to hero.</p>


<p align="center">
  <a href="#zero-to-hero">From zero to hero</a> •
  <a href="#installation">Installation</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributions">Contributions</a>
</p>


<h2 align="center">From zero to hero</h2>

**Texthero is a python toolkit that help you work with text-based dataset quickly and effortlessly**. Texthero is very simple to learn and designed to be used on top of Pandas.

You can think of Texthero as a tool to help you _understand_ and work with text-based dataset. Given a tabular dataset, it's easy to _grasp the main concept_. Instead, given a text dataset it's harder to have quick insights of the underline data. 

With Texthero, preprocessing text data, map it into vectors and visualize the obtained vector space takes only a couple of lines.

Texthero is composed of only three python modules *preprocessing.py*, *representation.py*, *visualization.py* and it's <a href="https://texthero.org/docs/getting-started">well documented</a>.

<h2 align="center">Installation</h2>

Install texthero via `pip`:

```bash
pip install texthero
```

> ☝️Under the hoods, Texthero makes use of multiple NLP and machine learning toolkits such as Gensim, NLTK, SpaCy and scikit-learn. You don't need to install them all separately, pip will take care of that.

<h2 align="center">Getting started</h2>

The best way to learn Texthero is through the <a href="https://texthero.org/docs/getting-started">Getting Started</a> docs. 

In case you are an advanced python user, then `help(texthero)` should do the work.

<h2 align="center">Example</h2>

<h3>1. Text cleaning, TF-IDF representation and visualization</h3>


```python
import texthero as hero
import pandas as pd

df = pd.read_csv(
   "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)

df['pca'] = (
   df['text']
   .pipe(hero.clean)
   .pipe(hero.tfidf)
   .pipe(hero.pca)
)
hero.scatterplot(df, 'pca', color='topic', title="PCA BBC Sport news")
```
<p align="center">
   <img src="https://github.com/jbesomi/texthero/raw/master/github/scatterplot_bbcsport.svg">
</p>

<h3>2. Text preprocessing, TF-IDF, K-means and visualization</h3>

```python
import texthero as hero
import pandas as pd

df = pd.read_csv(
    "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)

df['tfidf'] = (
    df['text']
    .pipe(hero.clean)
    .pipe(hero.tfidf)
)

df['kmeans_labels'] = (
    df['tfidf']
    .pipe(hero.kmeans, n_clusters=5)
    .astype(str)
)

df['pca'] = df['tfidf'].pipe(hero.pca)

hero.scatterplot(df, 'pca', color='kmeans_labels', title="K-means BBC Sport news")
```
<p align="center">
   <img src="https://github.com/jbesomi/texthero/raw/master/github/scatterplot_bbcsport_kmeans.svg">
</p>

<h3>3. Simple pipeline for text cleaning</h3>

You got some dirty text data

```
>>> import texthero as hero
>>> import pandas as pd
>>> text = "This sèntencé    (123 $) needs to [OK!] be cleaned!   "
>>> s = pd.Series(text)
>>> s
0    This sèntencé    (123 $) needs to [OK!] be cleane...
dtype: object
```

Remove all digits:

```
>>> s = hero.remove_digits(s)
>>> s
0    This sèntencé    (  $) needs to [OK!] be cleaned!
dtype: object
```

Remove all type of brackets and their content.

``` 
>>> s = hero.remove_brackets(s)
>>> s 
0    This sèntencé    needs to  be cleaned!
dtype: object
```

Remove diacritics.

``` 
>>> s = hero.remove_diacritics(s)
>>> s 
0    This sentence    needs to  be cleaned!
dtype: object
```

Remove punctuation.

``` 
>>> s = hero.remove_punctuation(s)
>>> s 
0    This sentence    needs to  be cleaned
dtype: object
```

Remove extra white-spaces.

``` 
>>> s = hero.remove_whitespace(s)
>>> s 
0    This sentence needs to be cleaned
dtype: object
```

Sometimes we also wants to get rid of stop-words.

```
>>> s = hero.remove_stopwords(s)
>>> s
0    This sentence needs cleaned
dtype: object
```

<h2 align="center">API</h2>

Texthero is composed of three modules: [preprocessing.py](/texthero/preprocessing.py), [representation.py](/texthero/representation.py) and [visualization.py](/texthero/visualization.py).

<h3>1. Preprocessing</h3>

**Scope:** prepare the **text** data for further analysis.

Full documentation: [preprocessing](https://texthero.org/docs/api-preprocessing)

<h3>2. Representation</h3>

**Scope:** map text data into vectors and do dimensionality reduction.

Supported **representation** algorithms:
1. Term frequency (`count`)
1. Term frequency-inverse document frequency (`tfidf`)

Supported **clustering** algorithms:
1. K-means (`kmeans`)
1. Density-Based Spatial Clustering of Applications with Noise (`dbscan`)
1. Meanshift (`meanshift`)

Supported **dimensionality reduction** algorithms:
1. Principal component analysis (`pca`)
1. t-distributed stochastic neighbor embedding (`tsne`)
1. Non-negative matrix factorization (`nmf`)

Full documentation: [representation](https://texthero.org/docs/api-representation)

<h3>3. Visualization</h3>

**Scope:** summarize the main facts regarding the text data and visualize it. This module is opinionable. It's handy for anyone that needs a quick solution to visualize on screen the text data, for instance during a text exploratory data analysis (EDA).

Supported functions:
   - Text scatterplot (`scatterplot`)
   - Most common words (`top_words`)

Full documentation: [visualization](https://texthero.org/docs/api-visualization)

<h2 align="center">Contributions</h2>

Pull requests are amazing and most welcome. Start by fork this repository and [open an issue](https://github.com/jbesomi/texthero/issues).

Texthero is also looking for maintainers and contributors. In case of interest, just drop a line at jonathanbesomi__AT__gmail.com
