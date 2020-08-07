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
   <a href="#from-zero-to-hero">From zero to hero</a> •
   <a href="#installation">Installation</a> •
   <a href="#getting-started">Getting Started</a> •
   <a href="#examples">Examples</a> •
   <a href="#api">API</a> •
   <a href="#faq">FAQ</a> •
   <a href="#contributions">Contributions</a>
</p>


<p align="center">
    <img src="https://github.com/jbesomi/texthero/raw/master/github/screencast.gif">
</p>

<h2 align="center">From zero to hero</h2>

Texthero is a python toolkit to work with text-based dataset quickly and effortlessly. Texthero is very simple to learn and designed to be used on top of Pandas. Texthero has the same expressiveness and power of Pandas and is extensively documented. Texthero is modern and conceived for programmers of the 2020 decade with little knowledge if any in linguistic. 

You can think of Texthero as a tool to help you _understand_ and work with text-based dataset. Given a tabular dataset, it's easy to _grasp the main concept_. Instead, given a text dataset, it's harder to have quick insights into the underline data. With Texthero, preprocessing text data, mapping it into vectors, and visualizing the obtained vector space takes just a couple of lines.

Texthero include tools for:
* Preprocess text data: it offers both out-of-the-box solutions but it's also flexible for custom-solutions.
* Natural Language Processing: keyphrases and keywords extraction, and named entity recognition.
* Text representation: TF-IDF, term frequency, and custom word-embeddings (wip)
* Vector space analysis: clustering (K-means, Meanshift, DBSCAN and Hierarchical), topic modeling (wip) and interpretation.
* Text visualization: vector space visualization, place localization on maps (wip).

Texthero is free, open-source and [well documented](https://texthero.org/docs) (and that's what we love most by the way!). 

We hope you will find pleasure working with Texthero as we had during his development.

<h2 align="center">Hablas español? क्या आप हिंदी बोलते हैं? 日本語が話せるのか？</h2>

Texthero has been developed for the whole NLP community. We know how hard it is to deal with different NLP tools (NLTK, SpaCy, Gensim, TextBlob, Sklearn): that's why we developed Texthero, to simplify things.

Now, the next main milestone is to provide *multilingual support* and for this big step, we need the help of all of you. ¿Hablas español? Sie sprechen Deutsch? 你会说中文？ 日本語が話せるのか？ Fala português? Parli Italiano? Вы говорите по-русски? If yes or you speak another language not mentioned here, then you might help us develop multilingual support! Even if you haven't contributed before or you just started with NLP, contact us or open a Github issue, there is always a first time :) We promise you will learn a lot, and, ... who knows? It might help you find your new job as an NLP-developer!

For improving the python toolkit and provide an even better experience, your aid and feedback are crucial. If you have any problem or suggestion please open a Github [issue](https://github.com/jbesomi/texthero/issues), we will be glad to support you and help you.


<h2 align="center">Beta version</h2>

Texthero's community is growing fast. Texthero though is still in a beta version; soon, a faster and better version will be released and it will bring some major changes.

For instance, to give a more granular control over the pipeline, starting from the next version on, all `preprocessing` functions will require as argument an already tokenized text. This will be a major change.

Once released the stable version (Texthero 2.0), backward compatibility will be respected. Until this point, backward compatibility will be present but it will be weaker.

If you want to be part of this fast-growing movements, do not hesitate to contribute: [CONTRIBUTING](./CONTRIBUTING.md)!

<h2 align="center">Installation</h2>

Install texthero via `pip`:

```bash
pip install texthero
```

> ☝️Under the hoods, Texthero makes use of multiple NLP and machine learning toolkits such as Gensim, NLTK, SpaCy and scikit-learn. You don't need to install them all separately, pip will take care of that.

> For faster performance, make sure you have installed Spacy version >= 2.2. Also, make sure you have a recent version of python, the higher, the best.

<h2 align="center">Getting started</h2>

The best way to learn Texthero is through the <a href="https://texthero.org/docs/getting-started">Getting Started</a> docs. 

In case you are an advanced python user, then `help(texthero)` should do the work.

<h2 align="center">Examples</h2>

<h3>1. Text cleaning, TF-IDF representation and Visualization</h3>


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

<h3>2. Text preprocessing, TF-IDF, K-means and Visualization</h3>

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

```python
>>> import texthero as hero
>>> import pandas as pd
>>> text = "This sèntencé    (123 /) needs to [OK!] be cleaned!   "
>>> s = pd.Series(text)
>>> s
0    This sèntencé    (123 /) needs to [OK!] be cleane...
dtype: object
```

Remove all digits:

```python
>>> s = hero.remove_digits(s)
>>> s
0    This sèntencé    (  /) needs to [OK!] be cleaned!
dtype: object
```

> Remove digits replaces only blocks of digits. The digits in the string "hello123" will not be removed. If we want to remove all digits, you need to set only_blocks to false.

Remove all types of brackets and their content.

```python
>>> s = hero.remove_brackets(s)
>>> s 
0    This sèntencé    needs to  be cleaned!
dtype: object
```

Remove diacritics.

```python
>>> s = hero.remove_diacritics(s)
>>> s 
0    This sentence    needs to  be cleaned!
dtype: object
```

Remove punctuation.

```python
>>> s = hero.remove_punctuation(s)
>>> s 
0    This sentence    needs to  be cleaned
dtype: object
```

Remove extra white-spaces.

```python
>>> s = hero.remove_whitespace(s)
>>> s 
0    This sentence needs to be cleaned
dtype: object
```

Sometimes we also want to get rid of stop-words.

```python
>>> s = hero.remove_stopwords(s)
>>> s
0    This sentence needs cleaned
dtype: object
```

<h2 align="center">API</h2>

Texthero is composed of four modules: [preprocessing.py](/texthero/preprocessing.py), [nlp.py](/texthero/nlp.py), [representation.py](/texthero/representation.py) and [visualization.py](/texthero/visualization.py).

<h3>1. Preprocessing</h3>

**Scope:** prepare **text** data for further analysis.

Full documentation: [preprocessing](https://texthero.org/docs/api-preprocessing)

<h3>2. NLP</h3>

**Scope:** provide classic natural language processing tools such as `named_entity` and `noun_phrases`.

Full documentation: [nlp](https://texthero.org/docs/api-nlp)


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

<h2 align="center">FAQ</h2>

<h5>Why Texthero</h5>

Sometimes we just want things done, right? Texthero helps with that. It helps make things easier and give the developer more time to focus on his custom requirements. We believe that cleaning text should just take a minute. Same for finding the most important part of a text and the same for representing it.

In a very pragmatic way, texthero has just one goal: make the developer spare time. Working with text data can be a pain and in most cases, a default pipeline can be quite good to start. There is always time to come back and improve previous work.


<h2 align="center">Contributions</h2>

> "Texthero has been developed by a member of the NLP community for the whole NLP-community"

Texthero is for all of us NLP-developers and it can continue to exist with the precious contribution of the community.

Your level of expertise of python and NLP does not matter, anyone can help and anyone is more than welcome to contribute!

**Are you an NLP expert?**

- [open an issue](https://github.com/jbesomi/texthero/issues) and tell us what you like and dislike of Texthero and what we can do better!

**Are you good at creating websites?**

The website will be soon moved from Docusaurus to Sphinx: read the [open issue there](https://github.com/jbesomi/texthero/issues/40). Good news: the website will look like now :) Average news: we need to do some web-development to adapt [this Sphinx template](https://github.com/jbesomi/texthero/issues/40) to our needs. Can you help us?

**Are you good at writing?**

Probably this is the most important piece missing now on Texthero: more tutorials and more "Getting Started" guide. 

If you are good at writing you can help us! Why don't you start by [Adding a FAQ page to the website](https://github.com/jbesomi/texthero/issues/41) or explain how to [create a custom pipeline](https://github.com/jbesomi/texthero/issues/38)? Need help? We are there for you.

**Are you good in python?**

There are a lot of [open issues](https://github.com/jbesomi/texthero/issues) for techie guys. Which one do you choose?

If you have just other questions or inquiry drop me a line at jonathanbesomi__AT__gmail.com

<h3>Contributors (in chronological order)</h3>

- [Selim Al Awwa](https://github.com/selimelawwa/)
- [Parth Gandhi](https://github.com/ParthGandhi)
- [Dan Keefe](https://github.com/Peritract)
- [Christian Claus](https://github.com/cclauss)
- [bobfang1992](https://github.com/bobfang1992)
- [Ishan Arora](https://github.com/ishanarora04)
- [Vidya P](https://github.com/vidyap-xgboost)
- [Cedric Conol](https://github.com/cedricconol)
- [Rich Ramalho](https://github.com/richecr)


<h2 align="center"><a href="./LICENSE">License</a></h2>

The MIT License (MIT)

Copyright (c) 2020 Texthero

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
