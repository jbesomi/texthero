---
id: getting-started-preprocessing
title: Getting started preprocessing
---

## Getting started with <span style="color: #ff8c42">preprocessing</span>

By now you should have a general overview of what Texthero is about, in the next sections we will dig a bit deeper into Texthero's core to appreciate its super powers when it comes to text data.

## Overview

Preprocessing is the stepping stone to any text analytics project, as well as one of Texthero's pillars. 

The Texthero's `clean` pipeline provides a great starting point to quickly implement standard preprocessing steps. If your project requires specific preprocessing steps, Texthero offers a `tool` to quickly experiment and find the best preprocessing solution. 

##### Preprocessing API

Check-out the complete [preprocessing API](/docs/api-preprocessing) for a detailed overview of Texthero's preprocessing functions. Texthero's approach to preprocessing is modular, allowing you maximum flexibility in customizing the preprocessing steps for your project.

##### Doing it right

There is no magic formula that fits all preprocessing needs. Texthero offers a modular and customizable approach ideal to preprocess data for bag-of-words models. 

> What is Bag-of-Words?
A bag-of-words model is a popular, simple and flexible way of extracting features from text for use in modeling, such as with machine learning algorithms. Feature extraction consists in converting text into numbers, specifically vectors of numbers, that a machine learning algorithm can read. A bag-of-words representation describe the occurrence of words within a document resorting on two elements:
1. A vocabulary of known words
2. A measure of presence of known words
For example, given the following two text documents:
```python
doc1 = "Hulk likes to eat avocados. Green Lantern likes avocados too."
doc2 = "Green Lantern also likes bonfires."
```
We can use bag-of-words representation to generate two dictionaries as follws:
```python
BoW1 = {"Hulk":1, "likes":2, "to":1, "eat":1, "avocados":2, "Green":1, "Lantern":1, "too":1}
BoW2 = {"Green":1, "Lantern":1, "also":1, "likes":1, "bonfires":1}
```
After transforming the text into a "bag of words", we can calculate various measures to characterize the text. The most common type of characteristics, or features, calculated from the bag-of-words model relates to term frequency, namely, the number of times a term appears in the text.

Texthero is a powerful tool to prepare data for bag-of-words modeling. It enables:
- Preliminary exploration of text data of any format and structure
- Extraction of relevant and clean content for use in bag-of-words models
- Flexibility in adapting to user-specific tasks and contexts

### Text preprocessing, From zero to hero

##### Standard pipeline

Let's see how Texthero can help with cleaning messy text data and get them ready for bag-of-words models.

```python
import texthero as hero
import pandas as pd
df = pd.DataFrame(
    ["I have the power! $$ (wow!)",
     "Flame on! <br>oh!</br>",
     "HULK SMASH!"], columns=['text'])
>>> df.head()
                           text
0  "I have the power! $$ (wow!)"  
1       "Flame on! <br>oh!</br>"  
2                  "HULK SMASH!"
```

To implement Texthero's standard preprocessing pipeline, it only takes one command:

```python
hero.preprocessing.clean(df['text'])
0         power wow
1    flame br oh br
2        hulk smash
Name: text, dtype: object
```

Texthero's `clean` pipeline takes as input the dataframe column containing the text to preprocess (df['text']) and returns a clean text series. For maximum compatibility with bag-of-words models, the standard cleaning process prioritizes pure text content over other aspects, such as grammar or puntuation. The text is cleaned from what is considered uninformative content, e.g. punctuation signs ("!", "()", ".", etc.), tags ("<br>", "</br>", etc.) and stopwords ("the", "on", etc.).

##### Custom pipeline

Assume that our project requires to keep all punctuation marks. For instance, because instead of bag-of-words we want to use a more advanced and complex neural network transformer where punctuation matters. 
We might still have specific preprocessing steps to implement. Such as the removal of all stand-alone content within round brackets.
Let's see how Texthero can help in this case...

The first step would be to search for the specific function that "removes content in parenthesis" in the [preprocessing API](/docs/api-preprocessing).
Turns out that "remove_round_brackets" is the function we are looking for as  it "removes content within brackets and the brackets itself".
We now need to create a custom preprocessing pipeline where the only implemented step is "remove_round_brackets". In order to do this, we resort on Pandas "pipe" function as follows:

```python
df['clean'] = (
    df['text']
    .pipe(hero.preprocessing.remove_round_brackets)
)
>>> df['clean'].head(2)
0      I have the power! $$ 
1    Flame on! <br>oh!</br>
Name: clean, dtype: object
```
The part of text within brackets "(wow!)" has been succesfully removed!


If our project required instead the removal of HTML tags only. We will proceed in a similar way:

```python
df['clean'] = (
    df['text']
    .pipe(hero.preprocessing.remove_html_tags)
)
>>> df['clean'].head(2)
0    I have the power! $$ (wow!)
1                  Flame on! oh!
Name: clean, dtype: object
```
The HTML tags "<br>" and "</br>" have now been removed!


If we were to apply both preprocessing steps above, the resulting custom pipeline will look like this:
```python
custom_pipeline = [hero.preprocessing.remove_round_brackets,
                   hero.preprocessing.remove_html_tags]
df['clean'] = hero.clean(df['text'], custom_pipeline)
```

##### Going further
If you are interested in learning more about text cleaning or NLP in general, check out these resources:

- Daniel Jurafsky and James H. Martin. 2008. Speech and Language Processing: An Introduction to Natural Language Processing, Speech Recognition, and Computational Linguistics. 2nd edition. Prentice-Hall.

- Christopher D. Manning and Hinrich Schütze. 1999. Foundations of Statistical Natural Language Processing. Cambridge, MA: MIT Press.