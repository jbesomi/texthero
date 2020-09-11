---
id: getting-started-preprocessing
---

## Getting started with <span style="color: #ff8c42">pre-processing</span>

Pre-processing is a fundamental step in text analysis. Being consistent and methodical in pre-processing operations is a necessary condition for the success of text-based analysis.

## Overview

--

## Intro

When we (as humans) read text from a book or a newspaper, the _input_ that our brain gets to allow us to understand that text is in the form of individual letters, that are then combined into words, sentences, paragraphs, etc... you get it!
The problem with having a machine reading text is simple: the machine doesn't know how to read letters, words, paragraphs, etc.
The machine however knows how to read numerical vectors and text has good properties that easily allow its conversion into a numerical representation. There are several sophisticated methods to make this conversion but, in order to perform well, all of them require that the text given as input is as clean and simple as possible, in other words **pre-processed**.
Clean and simple basically means eliminating any unnecessary information (e.g. the machine does not need to know about punctuation, page numbers or spacing) and solving as many ambiguities as possibe (so that, for instance, the verb "run" and its forms "ran", "runs", "running" will all refer to the same concept).

How useful is this step?
Have you ever heard the story that Data Scientists typically spend ~80% of their time to obtain a proper dataset and the remaining ~20% for actually using it? Well, for text is kind of the same thing. Pre-processing is a **fundamental step** in text analysis and it usually takes some time to be properly implemented.

In text hero it only takes one command:
To clean text data in a reliable way all we have to do is:
#Note for this section we use the same dataset as in **Getting Started**

```python
df['clean_text'] = hero.clean(df['text'])
```
or ...
[Pipeline explanation]

## Clean

Texthero clean method allows a rapid implementation of key cleaning steps that are:
- Derived from survey of relevant academic literature #cite
- Validated by a group of NLP enthusiasts with experience in applying these methods in different contexts #background
- Accepted by the NLP community as inescapable and standard

The default steps do the following:

#[TABLE]

in just one command:

```python
df['clean_text'] = hero.clean(df['text'])
```
## Custom Pipeline

Sometimes, project specificities might require different approach to pre-processing. For instance, you might decide that digits are important to your analyses if you are analyzing movies and one of them is "007-James Bond" or if you think that stopwords contain relevant information for your analysis setting.
If this is the case, you can easily edit the pre-processing pipeline by:
```python

```#Comment/explain what it does

If you are interested in learning more about text cleaning, check out these resources:
#Links list





## Customize it 

Let's see how texthero STANDARDIZE this step...


### Stemming

`do_stem` returns better results when used after `remove_punctuation`.

Example:

```python

>>> text = "I love climbing and running."
>>> hero .stem(pd.Series(text), stem="snowball")
   0    i love climb and running.
   dtype: object
```

Whereas 

```python

>>> text = "I love climbing and running"
>>> hero .stem(pd.Series(text), stem="snowball")
   0    i love climb and run
   dtype: object
```
