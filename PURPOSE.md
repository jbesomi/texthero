# PURPOSE

This document attempts at defining the purpose of Texthero and it's future enhancements.

### Motivation

We believe the text mining and text analytics community is missing a space for learning how to deal with all the different NLP/text mining/text analytics tools as well as a simple python package based on Pandas to work with text data effortlessly.

The lack is in a clear "Universal text mining and data analysis documentation" and that's the main purpose of **texthero**.

### Objective

We can decompose the objective of Texthero in two parts:

1. ** Offer an efficient tool to deal with text-based datasets (The texthero python package). Texthero is mainly a teaching tool and therefore easy to use and understand, but at the same time quite efficient and should be able to handle large quantities of data.

2. ** Provide a sustain to newcomers in the NLP world to efficiently learn all the main core topics (tf-idf, text cleaning, regular expression, etc). As there are many other tutorials, the main approach is to redirect users to valuable resources and explain better any missing point. This part is done mainly through the *tutorials* on texthero.org.


### Channels

1. **Github repository** development of texthero python package. The README should mainly discuss the PyPI package and not the extra tutorials.

2. **Texthero.org**
    The website acts both as the official documentation for the python package as well as a source of information to learn about how to deal with textual data.

    - **Getting Started** 4/5 pages document that explains how to use the Texthero tool. The tutorials assume a very basic understanding of the main topics (representation, tf-idf, word2vec, etc) but at the same time provide links to internal (tutorials) and external resources.

    - **Tutorials** Sort of a blog with articles related to NLP and text mining. This includes both tutorials on how to use certain texthero tools, how some part of the Texthero code has been developed as well as extra articles related to other parts of text analytics. Tutorials should focus on how to analyze large quantities of text.

    - **?** Open to any request. For ideas, open a new issue and/or contact jonathan.besomi__AT__gmail.com


### Python package

For future development, it is important to have a clear idea in mind of the purpose of Texthero as a python package.


**Package core purpose**

The goal is to extract insights from the whole corpora, i.e collection of document and not from the single element.

Generally, the corpora are composed of a __long__ collection of documents and therefore the required techniques need to be efficient to deal with a large amount of text.

**Neural network**

Texthero function (as of now) does not make use of a neural network solution. The main reason is that there is no need for that as there are mature libraries (PyTorch and Tensorflow to name a few).

What Texthero offers is a tool to be used in addition to any other machine learning libraries. Ideally, texthero should be used before applying any "sophisticated" approach to the dataset; to first better understand the underlying data before applying any complex model.


Note: a text corpus or collection of documents need to be always in form of a Pandas Series. "do that on a text corpus" or "do that on a Pandas Series" refers to the same act.

**Common usage**:
 - Clean a text Pandas Series
 - Tokenize a text Pandas Series
 - Represent a text Pandas Series
 - Benchmark on very simple models (Bayes ?) if changes improved the models
 - Understand a text without the need for using complex models such as Transformers.
 - Extract the main facts from a Pandas Series


**Naive Pandas Support**

Most of texthero python functions should accept as an argument a Pandas Series and return a Pandas Series. This permits to chain the different functions and also always append the Series to a Pandas Column.

Few exceptions:
    - When representing the data, the results might be very sparse, in this case, the returned value is a _Sparse_ Pandas Series. It's important to underline the difference in the documentation.

    - The "visualization" module might return visualization such as the count of top words. An alternative would be to add a custom `hero` accessor to access this kind of features.
    
