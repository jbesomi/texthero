"""
embeddings.py
-------------

Embed documents and work with the embeddings to find similar documents,
find topics, ... .

There are many different ways to transform text data into vectors to gain
insights from them. The resulting vectors are called _embeddings_.
A _word embedding_ assigns a vector to each word, a _document embedding_
(sometimes called thought _thought vector_)
assigns a vector to each document.

One way to get embeddings is to use a function such as tfidf,
count, or term_frequency that creates vectors depending
on which words occur how frequently.

Another option is to use embeddings that try to directly capture
the semantic relationship between words or sentences. For example,
the vector of 'biochemistry' minus the vector of 'chemistry' is
close to the vector of 'biology'.

In Texthero, both options are supported. The second option is
implemented through the `Flair library <https://github.com/flairNLP/flair>`_


TODO :
- think about:
    - Do we (want to) support word embeddings? How? Why?
    - Any opportunity for sparseness or alternatives to s.apply?
"""

import pandas as pd
import numpy as np

import flair
from flair.data import Sentence, Token

from typing import List, Union

from texthero._types import InputSeries, TokenSeries, VectorSeries


"""
Helper functions.
"""


def _texthero_init_for_flair_sentence(self, already_tokenized_text: List[str]):
    """
    To use flair embeddings, flair needs as input a
    'flair.Sentence' object. Creating such an object
    only works from strings in flair. However, we want
    our embeddings to work on TokenSeries, so we
    overwrite the 'flair Sentence' __init__ method with
    this method to create Sentence objects from already
    tokenized text.
    """

    super(Sentence, self).__init__()

    self.tokens: List[Token] = []
    self._embeddings: Dict = {}
    self.language_code: str = None
    self.tokenized = None

    # already tokenized -> simply add the tokens
    for token in already_tokenized_text:
        self.add_token(token)


# Overwrite flair Sentence __init__ method to handle already tokenized text
Sentence.__init__ = _texthero_init_for_flair_sentence


"""
Support for flair embeddings.
"""


@InputSeries(TokenSeries)
def embed(
    s: TokenSeries, flair_embedding: Union[WordEmbeddings, DocumentEmbeddings]
) -> VectorSeries:
    """
    TODO
    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> from flair.embeddings import TransformerDocumentEmbeddings
    >>> embedding = TransformerDocumentEmbeddings('bert-base-uncased')
    >>> s = pd.Series(["Text of doc 1", "Text of doc 2"]).pipe(hero.tokenize)
    >>> hero.embed(s, embedding)  # doctest: +SKIP
    0    [-0.6618074, -0.20467158, -0.05876905, -0.3482...
    1    [-0.5505255, -0.21915795, -0.0913163, -0.26856...
    dtype: object

    """

    def _embed_and_return_embedding(x):
        # flair embeddings need a 'flair Sentence' object as input.
        x = Sentence(x)
        # Calculate the embedding; flair writes it to x.embedding
        flair_embedding.embed(x)
        # Return it as numpy array.
        return x.embedding.detach().numpy()

    if isinstance(flair_embedding, DocumentEmbeddings):
        s = s.apply(lambda x: _embed_and_return_embedding(x))
    else:
        raise ValueError(
            "Unknown embedding type. Texthero only works with"
            " flair DocumentEmbeddings."
        )

    return s
