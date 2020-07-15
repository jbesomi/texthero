"""
Map words into vectors using different algorithms such as TF-IDF, word2vec or GloVe.
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix

from typing import Optional, Union, Any

import logging

logging.getLogger("gensim").setLevel(logging.WARNING)

from gensim.models import Word2Vec

from texthero import preprocessing

# from texthero import pandas_ as pd_

"""
Helper
"""


def representation_series_to_flat_series(
    s: Union[pd.Series, pd.Series.sparse],
    index: pd.Index = None,
    fill_nans_with: Any = np.nan,
) -> pd.Series:
    """
    Transform a Pandas Representation Series to a "normal" (flattened) Pandas Series.

    The given Series should have a multiindex with first level being the document
    and second level being individual features of that document (e.g. tdidf scores per word).
    The flattened Series has one cell per document, with the cell being a list of all
    the individual features of that document.

    Parameters
    ----------
    s : Sparse Pandas Series or Pandas Series
        The multiindexed Pandas Series to flatten.
    index : Pandas Index, optional, default to None
        The index the flattened Series should have.
    fill_nans_with : Any, default to np.nan
        Value to fill the NaNs (missing values) with.


    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> import numpy as np
    >>> index = pd.MultiIndex.from_tuples([("doc0", "Word1"), ("doc0", "Word3"), ("doc1", "Word2")], names=['document', 'word'])
    >>> s = pd.Series([3, np.nan, 4], index=index)
    >>> s
    document  word 
    doc0      Word1    3.0
              Word3    NaN
    doc1      Word2    4.0
    dtype: float64
    >>> hero.representation_series_to_flat_series(s, fill_nans_with=0.0)
    document
    doc0    [3.0, 0.0, 0.0]
    doc1    [0.0, 4.0, 0.0]
    dtype: object

    """
    s = s.unstack()

    if fill_nans_with is not np.nan:
        s.fillna(fill_nans_with, inplace=True)

    s = pd.Series(s.values.tolist(), index=s.index)

    if index is not None:
        s.index = index

    return s


# Warning message for not-tokenized inputs
_not_tokenized_warning_message = (
    "It seems like the given Pandas Series s is not tokenized. This function will"
    " tokenize it automatically using hero.tokenize(s) first. You should consider"
    " tokenizing it yourself first with hero.tokenize(s) in the future."
)


"""
Vectorization
"""


def term_frequency(
    s: pd.Series, max_features: Optional[int] = None, return_feature_names=False
):
    """
    Represent a text-based Pandas Series using term_frequency.

    The input Series should already be tokenized. If not, it will
    be tokenized before term_frequency is calculated.

    Parameters
    ----------
    s : Pandas Series
    max_features : int, optional
        Maximum number of features to keep.
    return_features_names : Boolean, False by Default
        If True, return a tuple (*term_frequency_series*, *features_names*)


    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"])
    >>> s = hero.tokenize(s)
    >>> hero.term_frequency(s)
    0    [1, 1, 0]
    1    [1, 0, 1]
    dtype: object
    
    To return the features_names:
    
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"])
    >>> s = hero.tokenize(s)
    >>> hero.term_frequency(s, return_feature_names=True)
    (0    [1, 1, 0]
    1    [1, 0, 1]
    dtype: object, ['Sentence', 'one', 'two'])

    """
    # TODO. Can be rewritten without sklearn.

    # Check if input is tokenized. Else, print warning and tokenize.
    if not isinstance(s.iloc[0], list):
        print(_not_tokenized_warning_message)
        s = preprocessing.tokenize(s)

    tf = CountVectorizer(
        max_features=max_features,
        lowercase=False,
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
    )
    s = pd.Series(tf.fit_transform(s).toarray().tolist(), index=s.index)

    if return_feature_names:
        return (s, tf.get_feature_names())
    else:
        return s


def tfidf(
    s: pd.Series, max_features=None, min_df=1, max_df=1.0, return_feature_names=False
) -> pd.Series.sparse:
    """
    Represent a text-based Pandas Series using TF-IDF.

    *Term Frequency - Inverse Document Frequency (TF-IDF)* is a formula to
    calculate the _relative importance_ of the words in a document, taking
    into account the words' occurences in other documents. It consists of two parts:

    The *term frequency (tf)* tells us how frequently a term is present in a document,
    so tf(document d, term t) = number of times t appears in d.

    The *inverse document frequency (idf)* measures how _important_ or _characteristic_
    a term is among the whole corpus (i.e. among all documents).
    Thus, idf(term t) = log((1 + number of documents) / (1 + number of documents where t is present)) + 1.

    Finally, tf-idf(document d, term t) = tf(d, t) * idf(t).

    The input Series should already be tokenized. If not, it will
    be tokenized before tfidf is calculated.

    Parameters
    ----------
    s : Pandas Series (tokenized)
    max_features : int, optional, default to 300
        Maximum number of features to keep.
    min_df : int, optional, default to 1.
        When building the vocabulary, ignore terms that have a document 
        frequency (number of documents a term appears in) strictly lower than the given threshold.
    max_df : int or double, optional, default to 1.0
        When building the vocabulary, ignore terms that have a document
        frequency (number of documents a term appears in) strictly higher than the given threshold. This arguments basically permits to remove corpus-specific stop words. When the argument is a float [0.0, 1.0], the parameter represents a proportion of documents.
    return_feature_names: Boolean, optional, default to False
        Whether to return the feature (i.e. word) names with the output.


    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Hi Bye", "Test Bye Bye"])
    >>> s = hero.tokenize(s)
    >>> hero.tfidf(s, return_feature_names=True)
    (document
    0    [1.0, 1.4054651081081644, 0.0]
    1    [2.0, 0.0, 1.4054651081081644]
    dtype: object, ['Bye', 'Hi', 'Test'])
    """

    # Check if input is tokenized. Else, print warning and tokenize.
    if not isinstance(s.iloc[0], list):
        print(_not_tokenized_warning_message)
        s = preprocessing.tokenize(s)

    tfidf = TfidfVectorizer(
        use_idf=True,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        norm=None,  # Disable l1/l2 normalization.
    )

    tfidf_vectors_csr = tfidf.fit_transform(s)

    # Result from sklearn is in Compressed Sparse Row format.
    # Pandas Sparse Series can only be initialized from Coordinate format.
    tfidf_vectors_coo = coo_matrix(tfidf_vectors_csr)
    s_out = pd.Series.sparse.from_coo(tfidf_vectors_coo)

    # Map word index to word name and keep original index of documents.
    feature_names = tfidf.get_feature_names()
    s_out.index = s_out.index.map(lambda x: (s.index[x[0]], feature_names[x[1]]))

    s_out.rename_axis(["document", "word"], inplace=True)

    # NOTE: Currently: still convert to flat series instead of representation series.
    # Will change to return representation series directly in Version 2.
    s_out = representation_series_to_flat_series(s_out, fill_nans_with=0.0)

    if return_feature_names:
        return s_out, feature_names
    else:
        return s_out


"""
Dimensionality reduction
"""


def pca(s, n_components=2):
    """
    Perform principal component analysis on the given Pandas Series.

    In general, *pca* should be called after the text has already been represented.

    Parameters
    ----------
    s : Pandas Series
    n_components : Int. Default is 2.
        Number of components to keep. If n_components is not set or None, all components are kept.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"])
 
    """
    pca = PCA(n_components=n_components)
    return pd.Series(pca.fit_transform(list(s)).tolist(), index=s.index)


def nmf(s, n_components=2):
    """
    Perform non-negative matrix factorization.

    
    """
    nmf = NMF(n_components=n_components, init="random", random_state=0)
    return pd.Series(nmf.fit_transform(list(s)).tolist(), index=s.index)


def tsne(
    s: pd.Series,
    n_components=2,
    perplexity=30.0,
    early_exaggeration=12.0,
    learning_rate=200.0,
    n_iter=1000,
    n_iter_without_progress=300,
    min_grad_norm=1e-07,
    metric="euclidean",
    init="random",
    verbose=0,
    random_state=None,
    method="barnes_hut",
    angle=0.5,
    n_jobs=-1,
):
    """
    Perform TSNE on the given pandas series.

    Parameters
    ----------
    s : Pandas Series
    n_components : int, default is 2.
        Number of components to keep. If n_components is not set or None, all components are kept.
    perplexity : int, default is 30.0

    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
        n_iter_without_progress=n_iter_without_progress,
        min_grad_norm=min_grad_norm,
        metric=metric,
        init=init,
        verbose=verbose,
        random_state=random_state,
        method=method,
        angle=angle,
        n_jobs=n_jobs,
    )
    return pd.Series(tsne.fit_transform(list(s)).tolist(), index=s.index)


"""
Clustering
"""


def kmeans(
    s: pd.Series,
    n_clusters=5,
    init="k-means++",
    n_init=10,
    max_iter=300,
    tol=0.0001,
    precompute_distances="auto",
    verbose=0,
    random_state=None,
    copy_x=True,
    n_jobs=-1,
    algorithm="auto",
):
    """
    Perform K-means clustering algorithm.
    """
    vectors = list(s)
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        precompute_distances=precompute_distances,
        verbose=verbose,
        random_state=random_state,
        copy_x=copy_x,
        n_jobs=n_jobs,
        algorithm=algorithm,
    ).fit(vectors)
    return pd.Series(kmeans.predict(vectors), index=s.index)


def dbscan(
    s,
    eps=0.5,
    min_samples=5,
    metric="euclidean",
    metric_params=None,
    algorithm="auto",
    leaf_size=30,
    p=None,
    n_jobs=None,
):
    """
    Perform DBSCAN clustering.
    """

    return pd.Series(
        DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        ).fit_predict(list(s)),
        index=s.index,
    )


def meanshift(
    s,
    bandwidth=None,
    seeds=None,
    bin_seeding=False,
    min_bin_freq=1,
    cluster_all=True,
    n_jobs=None,
    max_iter=300,
):
    """
    Perform mean shift clustering.
    """

    return pd.Series(
        MeanShift(
            bandwidth=bandwidth,
            seeds=seeds,
            bin_seeding=bin_seeding,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            n_jobs=n_jobs,
            max_iter=max_iter,
        ).fit_predict(list(s)),
        index=s.index,
    )


"""
Topic modelling
"""

# TODO.

"""
Word2Vec
"""


def word2vec(
    s: pd.Series,
    size=300,
    algorithm: str = "cbow",
    num_epochs: int = 30,
    min_count: int = 5,
    window_size: int = 5,
    alpha: float = 0.03,
    max_vocab_size: int = None,
    downsample_freq: float = 0.0001,
    min_alpha: float = 0.0001,
    negative_samples: int = 5,
    workers: int = None,
    seed: int = None,
):
    """Perform Word2vec on the given Pandas Series
    
    Return a Pandas Dataframe of shape (vocabulary_size, vectors_size).

    Word2vec is a two-layer neural network used to map each word to its vector representation. In general, its input is a text corpus and its output is a set of vectors: feature vectors that represent words in that corpus. In this specific case, the input is a Pandas Series containing in each cell a tokenized text and the output is a Pandas DataFrame where indexes are words and columns are the vector dimensions.

    Under the hoods, this function makes use of Gensim Word2Vec module.

    The input Series should already be tokenized. If not, it will
    be tokenized before word2vec is applied.

    
    Parameters
    ----------
    s : Pandas Series
    size : int, optional, default is 300
        Size of the returned vector. A good values is anything between 100-300. For very large dataset, a smaller values requires less training time.
    algorithm : str, optional, default is "cbow".
        The training algorithm. It can be either "skipgram" or "cbow". 
        With CBOW (continuous bag-of-words) the model predicts the current word from a window of surrounding context words. 
        In the continuous skip-gram mode, the model uses the current word to predict the surrounding window of context words.
        According to the authors, CBOW is faster while skip-gram is slower but does a better job for infrequent words.
    num_epochs : int, optional, default is 30
        Number of epochs to train the model.
    min_count : int, optional, default is 5
        Keep only words with a frequency equal or higher than min_count.
    window_size : int, optional, default is 5
        Surrounding window size of context words.
    alpha : float, optional, default is 0.03
        Initial learning rate
    max_vocab_size : int, optional, default to None
        Maximum number of words to keep. This corresponds to the length of the returned DataFrame. 
    downsample_freq : float, optional, default to 0.0001 (10^-4)
        Threshold frequency to downsample very frequent words. The results is similar to remove stop-words. The random removal of tokens is executed before word2vec is executed, reducing the distance between words. 
    min_alpha : float, default to 0.0001 (10^-4)
        The learning rate will drop linearly to min_alpha during training.
    negative_samples : int, optional, 5 by default
        Number of negative samples to use. Negative sampling addresses 
        the problem of avoding updating all weights at each epoch. It does so by selecting and modifing during each epoch only a small percentage of the total weights.

        The authors of the paper suggests to set negative sampling to 5-20 words for smaller datasets, and 2-5 words for large datasets.
    workers : int, optional, None by default.
        For improved performance, by default use all available computer workers. When set, use the same number of cpu.
    seed : int, optional, None by default.
        Seed for the random number generator. All vectors are initialized randomly using an hash function formed by the concatenation of the word itself and str(seed). Important: for a fully deterministically-reproducible run, you must set the model to run on a single worker thread (workers=1).

    See Also
    --------
    `Word2Vec Tutorial - The Skip-Gram Model <http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/>`_ and `Word2Vec Tutorial Part 2 - Negative Sampling <http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/>`_ for two great tutorial on Word2Vec

    """
    # Check if input is tokenized. Else, print warning and tokenize.
    if not isinstance(s.iloc[0], list):
        print(_not_tokenized_warning_message)
        s = preprocessing.tokenize(s)

    if algorithm == "cbow":
        sg = 0
    elif algorithm == "skipgram":
        sg = 1
    else:
        raise ValueError("algorithm must be either 'cbow' or 'skipgram'")

    w2v_model = Word2Vec(
        size=size,
        min_count=min_count,
        window=window_size,
        alpha=alpha,
        max_vocab_size=max_vocab_size,
        sample=downsample_freq,
        seed=seed,
        min_alpha=min_alpha,
        negative=negative_samples,
        sg=sg,
    )

    w2v_model.build_vocab(s.values, progress_per=10000)

    if len(w2v_model.wv.vocab.keys()) == 0:
        print("Vocabulary ...")

    w2v_model.train(
        s.values,
        total_examples=w2v_model.corpus_count,
        epochs=num_epochs,
        report_delay=1,
    )

    all_vocabulary = sorted(list(set(w2v_model.wv.vocab.keys())))

    return pd.DataFrame(data=w2v_model.wv[all_vocabulary], index=all_vocabulary)


def most_similar(df_embedding: pd.DataFrame, to: str) -> pd.Series:
    """
    Find most similar words to *to* for the given df_embedding.

    Given a Pandas DataFrame representing a word embedding, where each index is a word and the size of the dataframe corresponds to the length of the word vectors, return a Pandas Series containing as index the words and as value the cosine distance between *to* and the word itself.

    Parameters
    ----------
    df_embeddings: Pandas DataFrame
    to: str
        Word to find the most similar words to. That word must be in the DataFrame index.
    
    """

    if type(df_embedding) != pd.DataFrame:
        raise ValueError(
            "The first argument of most_similar must be a Pandas Dataframe representing a word embedding."
        )

    if to not in df_embedding.index:
        raise ValueError(
            f"Argument to={to} is not present in the index of the passed DataFrame."
        )

    return pd.Series(
        cosine_similarity(
            df_embedding, df_embedding.loc[to].to_numpy().reshape(1, -1)
        ).reshape(1, -1)[0],
        index=df_embedding.index,
    ).sort_values(ascending=True)
