---
id: getting-started-preprocessing
---

## Getting started with <span style="color: #ff8c42">pre-processing</span>

Pre-processing is a fundamental step in text analysis. Consistent, methodical and reproducible pre-processing operations are a necessary pre-requisite for success of any type of text-based analysis.


## Overview

When we (as humans) read text from a book or a newspaper, the _input_ that our brain gets to understand that text is in the form of individual letters, that are then combined into words, sentences, paragraphs, etc.
The problem with having a machine reading text is simple: the machine doesn't know how to read letters, words or paragraphs. The machine knows instead how to read _numerical vectors_. 
Text data has good properties that allow its conversion into a numerical representation. There are several sophisticated methods to make this conversion but, in order to perform well, all of them require the input text in a form that is as clean and simple as possible, in other words **pre-processed**.
Pre-processing text basically means eliminating any unnecessary information (e.g. the machine does not need to know about punctuation, page numbers or spacing between paragraphs) and solving as many ambiguities as possible (so that, for instance, the verb "run" and its forms "ran", "runs", "running" will all refer to the same concept).

How useful is this step?
Have you ever heard the story that Data Scientists typically spend ~80% of their time to obtain a proper dataset and the remaining ~20% to actually analyze it? Well, for text is kind of the same thing. Pre-processing is a **fundamental step** in text analysis and it usually takes some time to be properly and unambiguously implemented.

With text hero it only takes one command!
To clean text data in a reliable way all we have to do is:

```python
df['clean_text'] = hero.clean(df['text'])
```

> NOTE. In this section we use the same [BBC Sport Dataset](http://mlg.ucd.ie/datasets/bbc.html) as in **Getting Started**. To load the `bbc sport` dataset in a Pandas DataFrame run:
```python
df = pd.read_csv(
   "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)
```

## Key Functions

### Clean

Texthero's clean method allows a rapid implementation of key cleaning steps that are:

- Derived from review of relevant academic literature (#include citations)
- Validated by a group of NLP enthusiasts with applied experience in different contexts
- Accepted by the NLP community as standard and inescapable

The default steps do the following:

| Step                 | Description                                            |
|----------------------|--------------------------------------------------------|
|`fillna()`            |Replace missing values with empty spaces                |
|`lowercase()`         |Lowercase all text to make the analysis case-insensitive|
|`remove_digits()`     |Remove numbers                                          |
|`remove_punctuation()`|Remove punctuation symbols (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~) |
|`remove_diacritics()` |Remove accents
|`remove_stopwords()`  |Remove the most common words ("i", "me", "myself", "we", "our", etc.) |
                   
|`remove_whitespace()` |Remove spaces between words|



in just one command!

```python
df['clean_text'] = hero.clean(df['text'])
```

##### Custom Pipelines

Sometimes, project specificities might require different approaches to pre-processing. For instance, you might decide that digits are important to your analyses if you are analyzing movies and one of them is "007-James Bond". Or, you might decide that in your specific setting stopwords contain relevant information (e.g. if your data is about music bands and contains "The Who" or "Take That").
If this is the case, you can easily customize the pre-processing pipeline by implementing only specific cleaning steps:

```python
from texthero import preprocessing

custom_pipeline = [preprocessing.fillna,
                   preprocessing.lowercase,
                   preprocessing.remove_punctuation
                   preprocessing.remove_whitespace]
df['clean_text'] = hero.clean(df['text'], custom_pipeline)
```

or alternatively

```python
df['clean_text'] = df['clean_text'].pipe(hero.clean, custom_pipeline)
```

In the above example we want to pre-process the text despite keeping accents, digits and stop words.

### Tokenize

Given a character sequence, tokenization is the task of chopping it up into pieces, called tokens. Here is an example of tokenization:

Text: "Hulk is the greenest superhero!"
Tokens: "hulk", "is", "the", "greenest", "superhero", "!"

A token is a sequence of character grouped together as a useful semantic unit for processing. The major question of the tokenization step is how to make the split. In the example above it was quite straightforward: we chopped up the sentence on white spaces. But what would you do if the input text was:
"Hulk isn't the greenest superhero, Green Lantern is!"

Notice that the "isn't" contraction could lead to any of the following tokens:
"isnt", "isn't", "is" + "n't", "isn" + "t"

Tokenization issues are language specific and the process can involve ambiguity if tokens such as monetary amounts, numbers, hyphen-separated words or URLs are involved.

Texthero takes care of making the best set of choices based on the most reasonable assumptions...in just one command!

```python
from texthero import tokenize

s = pd.Series(["Hulk is the greenest superhero!"])
tokenize(s)
```

## Preprocessing API

Check-out the complete [preprocessing API](/docs/api-preprocessing) to discover how to customize the preprocessing steps according to your specific needs.


If you are interested in learning more about text cleaning or NLP in general, check out these resources:

- Daniel Jurafsky and James H. Martin. 2008. Speech and Language Processing: An Introduction to Natural Language Processing, Speech Recognition, and Computational Linguistics. 2nd edition. Prentice-Hall.

- Christopher D. Manning and Hinrich Schütze. 1999. Foundations of Statistical Natural Language Processing. Cambridge, MA: MIT Press.

