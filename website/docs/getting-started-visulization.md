# Visualisation

Now, before you start processing your dataset, you might want to visualize your data first in order to get the gist of the data and to choose, which NLP toolkit and models will be most suitable. The following tutorial will introduce two methods. Those will show you in a quick way the most frequent words in our dataset.

## 1. Top words


The easiest way to find out the most important, is, to have a look at
their absolute occurence in the set. This is simply how often a word/token occurs in your set. Before Texthero that easy task was quite complex to program. You first needed to write your own tokenizer, generate a DocumentTerm matrix with the CountVectorizer for example, then sum over one axis and sort in the end. This process is now simplyfied by Texthero.

```python
>>> # save the dataset in df
>>> import texthero as hero
>>> import pandas as pd
>>> df = pd.read_csv("https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv")
>>> # now we will extract all top words
>>> top_words = hero.top_words(df["text"])
>>> top_words.head()
the 12790
to 7051
a 5516
in 5271
and 5259
Name: text, dtype: int64
```

However, we can now see, that from the most common words we don't get so much information, as we have hoped for. This is, that the english language does not only contain relevant words with information, but also stopwords, which purpose is, to connect important words to gramaticaly complete sentences. To extract the most relevant parts of the texts, we will now first clean it with the texthero `clean` function and then look again at the topwords.

```python
>>> df["clean"] = hero.clean(df["text"])
>>> top_words = hero.top_words(df["clean"])
said 1338
first 790
england 749
game 681
one 671
Name: clean, dtype: int64
```

Now we can see, that for example most of the texts contain "england" and are about "games". That is now quite useful for further analysis and can be done in just two lines of code.

## 2. Wordcloud

But the data frame is still quite technical and less graphical. This can be improved by generating a WordCloud. Texthero has a build-in function, which calls the word_cloud package API to generate the picture. A wordcloud consits of the top words in our dataset arranged in a cloud, where the more frequent words as visualised bigger than the less frequent ones. When executing the following lines in a jupyter notebook, it will show you a wordcloud with the most common words
```python
>>> import texthero as hero
>>> import pandas as pd
>>> df = pd.read_csv("https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv")
>>> df["clean"] = hero.clean(df["text"])
>>> hero.wordcloud(df["clean"])
```

![](/img/wordcloud.png)

Here we can easily recognise the popular words from before, as they are printed bigger than the others.
