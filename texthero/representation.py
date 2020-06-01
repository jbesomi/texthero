"""
Map words into vectors using different algorithms such as TF-IDF, word2vec or GloVe.
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, DBSCAN, MeanShift

from typing import Optional

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
