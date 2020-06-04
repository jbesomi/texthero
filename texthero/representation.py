"""
Map words into vectors using different algorithms such as TF-IDF, word2vec or GloVe.
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics.pairwise import cosine_similarity

from typing import Optional

import logging

logging.getLogger("gensim").setLevel(logging.WARNING)

from gensim.models import Word2Vec

# from texthero import pandas_ as pd_
"""
Vectorization
"""


def term_frequency(
    s: pd.Series, max_features: Optional[int] = None, return_feature_names=False
):
    """
    Represent a text-based Pandas Series using term_frequency.

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
    >>> hero.term_frequency(s)
    0    [1, 1, 0]
    1    [1, 0, 1]
    dtype: object
    
    To return the features_names:
    
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"])
    >>> hero.term_frequency(s, return_feature_names=True)
    (0    [1, 1, 0]
    1    [1, 0, 1]
    dtype: object, ['Sentence', 'one', 'two'])

    """
    # TODO. Can be rewritten without sklearn.
    tf = CountVectorizer(
        max_features=max_features, lowercase=False, token_pattern="\S+"
    )
    s = pd.Series(tf.fit_transform(s).toarray().tolist(), index=s.index)

    if return_feature_names:
        return (s, tf.get_feature_names())
    else:
        return s


def tfidf(s: pd.Series, max_features=None, min_df=1, return_feature_names=False):
    """
    Represent a text-based Pandas Series using TF-IDF.

    Parameters
    ----------
    s : Pandas Series
    max_features : int, optional
        Maximum number of features to keep.
    min_df : int, optional. Default to 1.
        When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
    return_features_names : Boolean. Default to False.
        If True, return a tuple (*tfidf_series*, *features_names*)


    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"])
    >>> hero.tfidf(s)
    0    [0.5797386715376657, 0.8148024746671689, 0.0]
    1    [0.5797386715376657, 0.0, 0.8148024746671689]
    dtype: object
    
    To return the *feature_names*:
    
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"])
    >>> hero.tfidf(s, return_feature_names=True)
    (0    [0.5797386715376657, 0.8148024746671689, 0.0]
    1    [0.5797386715376657, 0.0, 0.8148024746671689]
    dtype: object, ['Sentence', 'one', 'two'])
    """

    # TODO. In docstring show formula to compute TF-IDF and also avoid using sk-learn if possible.

    tfidf = TfidfVectorizer(
        use_idf=True,
        max_features=max_features,
        min_df=min_df,
        token_pattern="\S+",
        lowercase=False,
    )
    s = pd.Series(tfidf.fit_transform(s).toarray().tolist(), index=s.index)

    if return_feature_names:
        return (s, tfidf.get_feature_names())
    else:
        return s


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
