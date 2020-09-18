"""
Map words into vectors using different algorithms such as TF-IDF, word2vec or GloVe.
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sklearn_normalize
from scipy.sparse import coo_matrix, csr_matrix, issparse

import pyLDAvis

from typing import Optional, Union, Any

from texthero import preprocessing

import logging
import warnings

# from texthero import pandas_ as pd_

"""
Helper
"""


def _check_is_valid_DocumentTermDF(df: Union[pd.DataFrame, pd.Series]) -> bool:
    """
    Check if the given Pandas Series is a Document Term DF.

    Returns true if input is Document Term DF, else False.

    """
    return isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex)


# Warning message for not-tokenized inputs
_not_tokenized_warning_message = (
    "It seems like the given Pandas Series s is not tokenized. This function will"
    " tokenize it automatically using hero.tokenize(s) first. You should consider"
    " tokenizing it yourself first with hero.tokenize(s) in the future."
)


"""
Vectorization
"""


def count(
    s: pd.Series,
    max_features: Optional[int] = None,
    min_df=1,
    max_df=1.0,
    binary=False,
) -> pd.DataFrame:
    """
    Represent a text-based Pandas Series using count.

    Return a Document Term DataFrame with the
    number of occurences of a document's words for every
    document.
    TODO add tutorial link

    The input Series should already be tokenized. If not, it will
    be tokenized before count is calculated.

    Parameters
    ----------
    s : Pandas Series (tokenized)

    max_features : int, optional, default to None.
        Maximum number of features to keep. Will keep all features if set to
        None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency (number of documents they appear in) strictly 
        lower than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        Ignore terms that have a document frequency (number of documents they
        appear in) frequency strictly higher than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    binary : bool, default=False
        If True, all non zero counts are set to 1.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"]).pipe(hero.tokenize)
    >>> hero.count(s) # doctest: +SKIP
         count        
      Sentence one two
    0        1   1   0
    1        1   0   1
   
    See Also
    --------

    Document Term DataFrame: TODO add tutorial link
    """
    # TODO. Can be rewritten without sklearn.

    # Check if input is tokenized. Else, print warning and tokenize.
    if not isinstance(s.iloc[0], list):
        warnings.warn(_not_tokenized_warning_message, DeprecationWarning)
        s = preprocessing.tokenize(s)

    tf = CountVectorizer(
        max_features=max_features,
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        min_df=min_df,
        max_df=max_df,
        binary=binary,
    )

    tf_vectors_csr = tf.fit_transform(s)

    multiindexed_columns = pd.MultiIndex.from_tuples(
        [("count", word) for word in tf.get_feature_names()]
    )

    return pd.DataFrame.sparse.from_spmatrix(
        tf_vectors_csr, s.index, multiindexed_columns
    )


def term_frequency(
    s: pd.Series, max_features: Optional[int] = None, min_df=1, max_df=1.0,
) -> pd.DataFrame:
    """
    Represent a text-based Pandas Series using term frequency.

    Return a Document Term DataFrame with the
    term frequencies of the terms for every
    document. The output is sparse.
    TODO add tutorial link

    The input Series should already be tokenized. If not, it will
    be tokenized before term_frequency is calculated.

    Parameters
    ----------
    s : Pandas Series (tokenized)

    max_features : int, optional, default to None.
        Maximum number of features to keep. Will keep all features if set to
        None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency (number of documents they appear in) strictly 
        lower than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        Ignore terms that have a document frequency (number of documents they
        appear in) frequency strictly higher than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one hey", "Sentence two"]).pipe(hero.tokenize)
    >>> hero.term_frequency(s) # doctest: +SKIP
      term_frequency               
            Sentence  hey  one  two
    0            0.2  0.2  0.2  0.0
    1            0.2  0.0  0.0  0.2

    See Also
    --------
    Document Term DataFrame: TODO add tutorial link
    """
    # Check if input is tokenized. Else, print warning and tokenize.
    if not isinstance(s.iloc[0], list):
        warnings.warn(_not_tokenized_warning_message, DeprecationWarning)
        s = preprocessing.tokenize(s)

    tf = CountVectorizer(
        max_features=max_features,
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        min_df=min_df,
        max_df=max_df,
    )

    tf_vectors_csr = tf.fit_transform(s)
    tf_vectors_coo = coo_matrix(tf_vectors_csr)

    total_count_coo = np.sum(tf_vectors_coo)
    frequency_coo = np.divide(tf_vectors_coo, total_count_coo)

    multiindexed_columns = pd.MultiIndex.from_tuples(
        [("term_frequency", word) for word in tf.get_feature_names()]
    )

    return pd.DataFrame.sparse.from_spmatrix(
        frequency_coo, s.index, multiindexed_columns
    )


def tfidf(s: pd.Series, max_features=None, min_df=1, max_df=1.0,) -> pd.DataFrame:
    """
    Represent a text-based Pandas Series using TF-IDF.

    *Term Frequency - Inverse Document Frequency (TF-IDF)* is a formula to
    calculate the _relative importance_ of the words in a document, taking
    into account the words' occurences in other documents. It consists of two
    parts:

    The *term frequency (tf)* tells us how frequently a term is present in a
    document, so tf(document d, term t) = number of times t appears in d.

    The *inverse document frequency (idf)* measures how _important_ or
    _characteristic_ a term is among the whole corpus (i.e. among all
    documents). Thus, idf(term t) = log((1 + number of documents) /
    (1 + number of documents where t is present)) + 1.

    Finally, tf-idf(document d, term t) = tf(d, t) * idf(t).

    Different from the `sklearn-implementation of 
    tfidf <https://scikit-learn.org/stable/modules/generated/sklearn.feature_
    extraction.text.TfidfVectorizer.html>`, this function does *not* normalize
    the output in any way, so the result is exactly what you get applying the
    formula described above.

    Return a Document Term DataFrame with the
    tfidf of every word in the document. The output is sparse.
    TODO add tutorial link

    The input Series should already be tokenized. If not, it will
    be tokenized before tfidf is calculated.

    Parameters
    ----------
    s : Pandas Series (tokenized)

    max_features : int, optional, default to None.
        If not None, only the max_features most frequent tokens are used.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency (number of documents they appear in) strictly 
        lower than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        Ignore terms that have a document frequency (number of documents they
        appear in) frequency strictly higher than the given threshold.
        This arguments basically permits to remove corpus-specific stop words.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Hi Bye", "Test Bye Bye"]).pipe(hero.tokenize)
    >>> hero.tfidf(s) # doctest: +SKIP
      tfidf                    
        Bye        Hi      Test
    0   1.0  1.405465  0.000000
    1   2.0  0.000000  1.405465

    See Also
    --------
    `TF-IDF on Wikipedia <https://en.wikipedia.org/wiki/Tf-idf>`_

    Document Term DataFrame: TODO add tutorial link
    """

    # Check if input is tokenized. Else, print warning and tokenize.
    if not isinstance(s.iloc[0], list):
        warnings.warn(_not_tokenized_warning_message, DeprecationWarning)
        s = preprocessing.tokenize(s)

    tfidf = TfidfVectorizer(
        use_idf=True,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        norm=None,  # Disable l1/l2 normalization.
    )

    tfidf_vectors_csr = tfidf.fit_transform(s)

    multiindexed_columns = pd.MultiIndex.from_tuples(
        [("tfidf", word) for word in tfidf.get_feature_names()]
    )

    return pd.DataFrame.sparse.from_spmatrix(
        tfidf_vectors_csr, s.index, multiindexed_columns
    )


"""
Dimensionality reduction
"""


def pca(
    s: Union[pd.Series, pd.DataFrame], n_components=2, random_state=None
) -> pd.Series:
    """
    Perform principal component analysis on the given Pandas Series.

    Principal Component Analysis (PCA) is a statistical method that is used
    to reveal where the variance in a dataset comes from. For textual data,
    one could for example first represent a Series of documents using
    :meth:`texthero.representation.tfidf` to get a vector representation of
    each document. Then, PCA can generate new vectors from the tfidf
    representation that showcase the differences among the documents most
    strongly in fewer dimensions.

    For example, the tfidf vectors will have length 100 if hero.tfidf was
    called on a large corpus with max_features=100. Visualizing 100 dimensions
    is hard! Using PCA with n_components=3, every document will now get a
    vector of length 3, and the vectors will be chosen so that the document
    differences are easily visible. The corpus can now be visualized in 3D and
    we can get a good first view of the data!

    In general, *pca* should be called after the text has already been
    represented to a matrix form.

    PCA cannot directly handle sparse input, so when calling pca on a
    DocumentTermDF, the input has to be expanded which can lead to
    memory problems with big datasets.

    Parameters
    ----------
    s : Pandas Series (VectorSeries) or MultiIndex Sparse DataFrame (DocumentTermDF)

    n_components : Int. Default is 2.
        Number of components to keep (dimensionality of output vectors).
        If n_components is not set or None, all components are kept.

    random_state : int, default=None
        Pass an int for reproducible results across multiple function calls.


    Returns
    -------
    Pandas Series with the vector calculated by PCA for the document in every
    cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football is great",
    ...                "Hi, I'm Texthero, who are you? Tell me!"])
    >>> s = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> # Attention, your results might differ due to
    >>> # the randomness in PCA!
    >>> hero.pca(s) # doctest: +SKIP
    document
    0     [1.5713577608669735, 1.1102230246251565e-16]
    1    [-1.5713577608669729, 1.1102230246251568e-16]
    dtype: object

    See also
    --------
    `PCA on Wikipedia <https://en.wikipedia.org/wiki/Principal_component_analysis>`_

    """
    pca = PCA(n_components=n_components, random_state=random_state, copy=False)

    if _check_is_valid_DocumentTermDF(s):
        values = s.values
    else:
        values = list(s)

    return pd.Series(list(pca.fit_transform(values)), index=s.index)


def nmf(
    s: Union[pd.Series, pd.DataFrame], n_components=2, random_state=None
) -> pd.Series:
    """
    Performs non-negative matrix factorization.

    Non-Negative Matrix Factorization (NMF) is often used in
    natural language processing to find clusters of similar
    texts (e.g. some texts in a corpus might be about sports
    and some about music, so they will differ in the usage
    of technical terms; see the example below). 

    Given a document-term matrix (so in
    texthero usually a Series after applying
    :meth:`texthero.representation.tfidf` or some other first representation
    function that assigns a scalar (a weight) to each word), NMF will find
    n_components many topics (clusters) and calculate a vector for each
    document that places it correctly among the topics.

    NMF can directly handle sparse input, so when calling nmf on a
    DocumentTermDF, the advantage of sparseness is kept.

    Parameters
    ----------
    s : Pandas Series (VectorSeries) or MultiIndex Sparse DataFrame (DocumentTermDF)

    n_components : Int. Default is 2.
        Number of components to keep (dimensionality of output vectors).
        If n_components is not set or None, all components are kept.

    random_state : int, default=None
        Pass an int for reproducible results across multiple function calls.

    Returns
    -------
    Pandas Series with the vector calculated by NMF for the document in every
    cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "Music, Violin, Orchestra",
    ...                "Football, Music"])
    >>> s = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.term_frequency)
    >>> hero.nmf(s) # doctest: +SKIP
    0                    [0.9080190347553924, 0.0]
    1                     [0.0, 0.771931061231598]
    2    [0.3725409073202516, 0.31656880119331093]
    dtype: object
    >>> # As we can see, the third document, which
    >>> # is a mix of sports and music, is placed
    >>> # between the two axes (the topics) while
    >>> # the other documents are placed right on 
    >>> # one topic axis each.

    See also
    --------
    `NMF on Wikipedia
    <https://en.wikipedia.org/wiki/Non-negative_matrix_factorization>`_

    """
    nmf = NMF(n_components=n_components, init="random", random_state=random_state,)

    if _check_is_valid_DocumentTermDF(s):
        s_coo = s.sparse.to_coo()
        s_for_vectorization = s_coo.astype("float64")
    else:
        s_for_vectorization = list(s)

    return pd.Series(list(nmf.fit_transform(s_for_vectorization)), index=s.index)


def tsne(
    s: Union[pd.Series, pd.DataFrame],
    n_components=2,
    perplexity=30.0,
    learning_rate=200.0,
    n_iter=1000,
    random_state=None,
    n_jobs=-1,
) -> pd.Series:
    """
    Performs TSNE on the given pandas series.

    t-distributed Stochastic Neighbor Embedding (t-SNE) is
    a machine learning algorithm used to visualize high-dimensional data in
    fewer dimensions. In natural language processing, the high-dimensional data
    is usually a document-term matrix (so in texthero usually a Series after
    applying :meth:`texthero.representation.tfidf` or some other first
    representation function that assigns a scalar (a weight) to each word)
    that is hard to visualize as there might be many terms. With t-SNE, every
    document gets a new, low-dimensional (n_components entries) vector in such
    a way that the differences / similarities between documents are preserved.

    T-SNE can directly handle sparse input, so when calling tsne on a
    DocumentTermDF, the advantage of sparseness is kept.

    Parameters
    ----------
    s : Pandas Series (VectorSeries) or MultiIndex Sparse DataFrame (DocumentTermDF)

    n_components : int, default is 2.
        Number of components to keep (dimensionality of output vectors).
        If n_components is not set or None, all components are kept.

    perplexity : float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significanlty
        different results.

    learning_rate : float, optional (default: 200.0)
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.

    n_iter : int, optional (default: 1000)
        Maximum number of iterations for the optimization. Should be at
        least 250.

    random_state : int, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls.

    n_jobs : int, optional, default=-1
        The number of parallel jobs to run for neighbors search.
        ``-1`` means using all processors.

    Returns
    -------
    Pandas Series with the vector calculated by t-SNE for the document in every
    cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "Music, Violin, Orchestra",
    ...                "Football, Music"])
    >>> s = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.term_frequency)
    >>> hero.tsne(s, random_state=42) # doctest: +SKIP
    0      [-18.833383560180664, -276.800537109375]
    1     [-210.60179138183594, 143.00535583496094]
    2    [-478.27984619140625, -232.97410583496094]
    dtype: object

    See also
    --------
    `t-SNE on Wikipedia <https://en.wikipedia.org/wiki/T-distributed_
    stochastic_neighbor_embedding>`_

    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    if _check_is_valid_DocumentTermDF(s):
        s_coo = s.sparse.to_coo()
        s_for_vectorization = s_coo.astype("float64")
    else:
        s_for_vectorization = list(s)

    return pd.Series(list(tsne.fit_transform(s_for_vectorization)), index=s.index)


"""
Clustering
"""


def kmeans(
    s: Union[pd.Series, pd.DataFrame],
    n_clusters=5,
    n_init=10,
    max_iter=300,
    random_state=None,
    algorithm="auto",
):
    """
    Performs K-means clustering algorithm.

    K-means clustering is used in natural language processing
    to separate texts into k clusters (groups) 
    (e.g. some texts in a corpus might be about sports
    and some about music, so they will differ in the usage
    of technical terms; the K-means algorithm uses this
    to separate them into two clusters). 

    Given a document-term matrix (so in
    texthero usually a Series after applying
    :meth:`texthero.representation.tfidf` or some other first representation
    function that assigns a scalar (a weight) to each word), K-means will find
    k topics (clusters) and assign a topic to each document.

    Kmeans can directly handle sparse input, so when calling kmeans on a
    DocumentTermDF, the advantage of sparseness is kept.

    Parameters
    ----------
    s: Pandas Series (VectorSeries) or MultiIndex Sparse DataFrame (DocumentTermDF)

    n_clusters: Int, default to 5.
        The number of clusters to separate the data into.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    random_state : int, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    algorithm : {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient on data with well-defined
        clusters, by using the triangle inequality. However it's more memory
        intensive.

    Returns
    -------
    Pandas Series with the cluster the document was assigned to in each cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, fun, guitar"])
    >>> s = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.term_frequency)
    >>> hero.kmeans(s, n_clusters=2, random_state=42)
    0    1
    1    0
    2    1
    3    0
    dtype: category
    Categories (2, int64): [0, 1]
    >>> # As we can see, the documents are correctly
    >>> # separated into topics / clusters by the algorithm.

    See also
    --------
    `kmeans on Wikipedia <https://en.wikipedia.org/wiki/K-means_clustering>`_

    """

    if _check_is_valid_DocumentTermDF(s):
        s_coo = s.sparse.to_coo()
        s_for_vectorization = s_coo.astype("float64")
    else:
        s_for_vectorization = list(s)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        copy_x=True,
        algorithm=algorithm,
    ).fit(s_for_vectorization)
    return pd.Series(kmeans.predict(s_for_vectorization), index=s.index).astype(
        "category"
    )


def dbscan(
    s: Union[pd.Series, pd.DataFrame],
    eps=0.5,
    min_samples=5,
    metric="euclidean",
    metric_params=None,
    leaf_size=30,
    n_jobs=-1,
):
    """
    Perform DBSCAN clustering.

    Density-based spatial clustering of applications with noise (DBSCAN)
    is used in natural language processing
    to separate texts into clusters (groups)
    (e.g. some texts in a corpus might be about sports
    and some about music, so they will differ in the usage
    of technical terms; the DBSCAN algorithm uses this
    to separate them into clusters). It chooses the
    number of clusters on its own.

    Given a document-term matrix (so in
    texthero usually a Series after applying
    :meth:`texthero.representation.tfidf` or some other first representation
    function that assigns a scalar (a weight) to each word), DBSCAN will find
    topics (clusters) and assign a topic to each document.

    DBSCAN can directly handle sparse input, so when calling dbscan on a
    DocumentTermDF, the advantage of sparseness is kept.

    Parameters
    ----------
    s: Pandas Series (VectorSeries) or MultiIndex Sparse DataFrame (DocumentTermDF)

    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : string, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. Use `sorted(sklearn.neighbors.VALID_METRICS['brute'])`
        to see valid options.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    n_jobs : int, default=-1
        The number of parallel jobs to run.
        ``-1`` means using all processors.

    Returns
    -------
    Pandas Series with the cluster the document was assigned to in each cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, enjoy, guitar"])
    >>> s = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> hero.dbscan(s, min_samples=1, eps=4)
    0    0
    1    1
    2    0
    3    1
    dtype: category
    Categories (2, int64): [0, 1]
    >>> # As we can see, the documents are correctly
    >>> # separated into topics / clusters by the algorithm
    >>> # and we didn't even have to say how many topics there are!

    See also
    --------
    `DBSCAN on Wikipedia <https://en.wikipedia.org/wiki/DBSCAN>`_

    """

    if _check_is_valid_DocumentTermDF(s):
        s_coo = s.sparse.to_coo()
        s_for_vectorization = s_coo.astype("float64")
    else:
        s_for_vectorization = list(s)

    return pd.Series(
        DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
        ).fit_predict(s_for_vectorization),
        index=s.index,
    ).astype("category")


def meanshift(
    s: Union[pd.Series, pd.DataFrame],
    bandwidth=None,
    bin_seeding=False,
    min_bin_freq=1,
    cluster_all=True,
    n_jobs=-1,
    max_iter=300,
):
    """
    Perform mean shift clustering.

    Mean shift clustering
    is used in natural language processing
    to separate texts into clusters (groups)
    (e.g. some texts in a corpus might be about sports
    and some about music, so they will differ in the usage
    of technical terms; the mean shift algorithm uses this
    to separate them into clusters). It chooses the
    number of clusters on its own.

    Given a document-term matrix (so in
    texthero usually a Series after applying
    :meth:`texthero.representation.tfidf` or some other first representation
    function that assigns a scalar (a weight) to each word), mean shift will
    find topics (clusters) and assign a topic to each document.

    Menashift cannot directly handle sparse input, so when calling meanshift on a
    DocumentTermDF, the input has to be expanded which can lead to
    memory problems with big datasets.

    Parameters
    ----------
    s: Pandas Series (VectorSeries) or MultiIndex Sparse DataFrame (DocumentTermDF)

    bandwidth : float, default=None
        Bandwidth used in the RBF kernel.

        If not given, the bandwidth is estimated.
        Estimating takes time at least quadratic in the number of samples
        (i.e. documents). For large datasets, it’s wise to set the bandwidth
        to a small value.

    bin_seeding : bool, default=False
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.

    min_bin_freq : int, default=1
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds.

    cluster_all : bool, default=True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    n_jobs : int, default=-1
        The number of jobs to use for the computation.
        ``-1`` means using all processors

    max_iter : int, default=300
        Maximum number of iterations, per seed point before the clustering
        operation terminates (for that seed point), if has not converged yet.

    Returns
    -------
    Pandas Series with the cluster the document was assigned to in each cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
    >>> hero.meanshift(s, bandwidth=2)
    0    1
    1    1
    2    1
    3    0
    4    0
    5    0
    dtype: category
    Categories (2, int64): [0, 1]

    See also
    --------
    `Mean-Shift on Wikipedia <https://en.wikipedia.org/wiki/Mean_shift>`_

    """

    if _check_is_valid_DocumentTermDF(s):
        vectors = s.values
    else:
        vectors = list(s)

    return pd.Series(
        MeanShift(
            bandwidth=bandwidth,
            bin_seeding=bin_seeding,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            n_jobs=n_jobs,
            max_iter=max_iter,
        ).fit_predict(vectors),
        index=s.index,
    ).astype("category")


"""
Topic modelling
"""


def truncatedSVD(
    s: Union[pd.Series, pd.DataFrame], n_components=2, n_iter=5, random_state=None,
) -> pd.Series:
    """
    Perform TruncatedSVD on the given pandas Series.

    TruncatedSVD is an algorithmn which can be used to reduce the dimensions
    of a given series. In natural language processing, the high-dimensional data
    is usually a document-term matrix (so in texthero usually a Series after
    applying :meth:`texthero.representation.tfidf` or some other first
    representation function that assigns a scalar (a weight) to each word).
    This is used as a tool to extract the most important topics and words
    of a given Series. In this context it is also referred to as
    Latent Semantic Analysis (LSA) or Latent Semantic Indexing (LSI).

    TruncatedSVD can directly handle sparse input, so when calling truncatedSVD on a
    DocumentTermDF, the advantage of sparseness is kept.

    Parameters
    ----------
    s : Pandas Series (VectorSeries) or MultiIndex Sparse DataFrame (DocumentTermDF)

    n_components : int, default is 2.
        Number of components to keep (dimensionality of output vectors).
        When using truncatedSVD for Topic Modelling, this needs to be
        the number of topics.

    n_iter : int, optional (default: 5)
       Number of iterations for randomized SVD solver.

    random_state : int, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls.


    Returns
    -------
    Pandas Series with the vector calculated by truncadedSVD for the document in every
    cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "Music, Violin, Orchestra",
    ...                "Football, Music"])                
    >>> s = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.term_frequency)
    >>> hero.truncatedSVD(s, random_state=42) # doctest: +SKIP
    0      [0.14433756729740624, 0.15309310892394884]
    1      [0.14433756729740663, -0.1530931089239484]
    2    [0.14433756729740646, 7.211110073938366e-17]
    dtype: object

    See also
    --------
    `truncatedSVD on Wikipedia <https://en.wikipedia.org/wiki/Singular_value_decomposition#Truncated_SVD>`_

    """
    truncatedSVD = TruncatedSVD(
        n_components=n_components, n_iter=n_iter, random_state=random_state
    )

    if _check_is_valid_DocumentTermDF(s):
        s_coo = s.sparse.to_coo()
        s_for_vectorization = s_coo.astype("float64")
    else:
        s_for_vectorization = list(s)

    result = pd.Series(
        list(truncatedSVD.fit_transform(s_for_vectorization)), index=s.index
    )

    return result


def lda(
    s: Union[pd.Series, pd.DataFrame],
    n_components=10,
    max_iter=10,
    random_state=None,
    n_jobs=-1,
) -> pd.Series:
    """
    Performs Latent Dirichlet Allocation on the given Pandas Series
    or DataFrame.

    Latent Dirichlet Allocation (LDA) is a topic modeling algorithm 
    based on Dirichlet distribution. In natural language processing
    LDA is often used to categorize documents into different topics
    and generate top words from these topics. In this process LDA is
    used in combination with algorithms which generate document-term-
    matrices, like :meth:`count`, :meth:`tfidf` or :meth:`term_frequency`.

    LDA can directly handle sparse input, so when calling LDA on a
    sparse DataFrame, the advantage of sparseness is kept.

    Parameters
    ----------
    s : pd.Series (VectorSeries) or Sparse pd.DataFrame

    n_components : int, default is 10.
        Number of components to keep (dimensionality of output vectors).
        When using truncatedSVD for Topic Modelling, this needs to be
        the number of topics.

    max_iter : int, optional (default: 10)
        The maximum number of iterations. In each interation,
        the algorithm gets closer to convergence. Set this higher
        for potentially better results, but also longer runtime.

    random_state : int, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls.

    Returns
    -------
    Pandas Series (VectorSeries) with the vector calculated by LDA
    for the document in every cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "Music, Violin, Orchestra",
    ...                "Football, Music"])                
    >>> s = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.term_frequency)
    >>> hero.lda(s, random_state=42) # doctest: +SKIP
    0    [0.07272782580722714, 0.0727702366844115, 0.07...
    1    [0.07272782580700803, 0.07277023650761331, 0.0...
    2    [0.08000075593366586, 0.27990110380876265, 0.0...
    dtype: object

    See also
    --------
    `LDA on Wikipedia <https://de.wikipedia.org/wiki/Latent_Dirichlet_Allocation`_

    """

    lda = LatentDirichletAllocation(
        n_components=n_components, max_iter=max_iter, random_state=random_state
    )

    if _check_is_valid_DocumentTermDF(s):
        s_coo = s.sparse.to_coo()
        s_for_vectorization = s_coo.astype("float64")
    else:
        s_for_vectorization = list(s)

    result = pd.Series(list(lda.fit_transform(s_for_vectorization)), index=s.index)

    return result


def topics_from_topic_model(s_document_topic: pd.Series) -> pd.Series:
    # TODO: add types everywhere when they're merged
    """
    Find the topics from a topic model. Input has
    to be output of one of
    - :meth:`texthero.representation.lda`
    - :meth:`texthero.representation.truncatedSVD`,
    so the output of one of Texthero's Topic Modelling
    functions that returns a relation
    between documents and topics.

    The function uses the given relation of
    documents to topics to calculate the
    best-matching topic per document and
    returns a Series with the topic IDs.

    Parameters
    ----------
    s_document_topic: pd.Series

    One of 
    - :meth:`texthero.representation.lda`
    - :meth:`texthero.representation.truncatedSVD`,


    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, band, guitar"])
    >>> s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> # Use Latent Dirichlet Allocation to relate documents to topics.s
    >>> s_lda = s_tfidf.pipe(hero.lda, n_components=2)
    >>> # Extract the best-matching topic per document.
    >>> hero.topics_from_topic_model(s_lda) # doctest: +SKIP
    0    1
    1    0
    2    1
    3    0
    dtype: category
    Categories (2, int64): [0, 1]


    See Also
    --------
    TODO add tutorial link

    :meth:`texthero.visualization.top_words_per_topic`_ to find the top words
    per topic after applying this function.

    """

    document_topic_matrix = np.matrix(s_document_topic.tolist())

    # The document_topic_matrix relates documents to topics,
    # so it shows for each document (so for each row), how
    # strongly that document belongs to a topic. So
    # document_topic_matrix[X][Y] = how strongly document X belongs to topic Y.
    # We use argmax to find the index of the topic that a document
    # belongs most strongly to for each document (so for each row).
    # E.g. when the first row of the document_topic_matrix is
    # [0.2, 0.1, 0.2, 0.5], then the first document will be put into
    # topic / cluster 3 as the third entry (counting from 0) is
    # the best matching topic.
    cluster_IDs = np.argmax(document_topic_matrix, axis=1).getA1()

    return pd.Series(cluster_IDs, index=s_document_topic.index, dtype="category")


def topic_matrices(
    s_document_term: pd.DataFrame, s_document_topic: pd.Series,
):
    # TODO: add Hero types everywhere when they're merged
    """
    Get a DocumentTopic Matrix and a TopicTerm Matrix (both as Dataframes)
    from a DocumentTerm Matrix and a DocumentTopic Matrix.

    Recieves as first argument s_document_term, which is the
    output of one of 
    - :meth:`texthero.representation.tfidf`
    - :meth:`texthero.representation.count`
    - :meth:`texthero.representation.term_frequency`.

    Recieves as second argument s_document_topic, which is either
    the output of a clustering function
    or the output of a topic modelling function,
    so one of
    - :meth:`texthero.representation.kmeans`
    - :meth:`texthero.representation.dbscan`
    - :meth:`texthero.representation.meanshift`
    - :meth:`texthero.representation.lda`.

    Both these matrices (the first one relating documents to
    terms and the second one relating documents to topics)
    are used to generate a DocumentTopic Matrix
    (relating documents to topics) and a
    TopicTerm Matrix (relating topics to terms).

    When the second argument is the output of a clustering
    function, we create the document_topic_matrix
    through the cluster-IDs. So if document X is in cluster Y,
    then document_topic_matrix[X][Y] = 1.

    For example, when
    `s_document_topic = pd.Series([0, 2, 2, 1], dtype="category")`,
    then the document_topic_matrix is
    ```python
    1 0 0
    0 0 1
    0 0 1
    0 1 0
    ```

    When the second argument is the output of a topic modelling function,
    their output is already the document_topic_matrix that relates
    documents to topics.

    We then have in both cases the DocumentTerm Matrix and the DocumentTopic Matrix.
    We then get the TopicTerm Matrix through
    topic_term_matrix = document_term_matrix.T * document_topic_matrix.

    Parameters
    ----------
    s_document_term : pd.DataFrame
        Output of one of
        :meth:`texthero.representation.tfidf`,
        :meth:`texthero.representation.count`,
        :meth:`texthero.representation.term_frequency`.

    s_document_topic : pd.Series
        Output of one of
        :meth:`texthero.representation.kmeans`,
        :meth:`texthero.representation.dbscan`,
        :meth:`texthero.representation.meanshift`,
        :meth:`texthero.representation.lda`.

    Returns
    -------
    Tuple of DataFrames.

    First one is
    DocumentTopic DataFrame where the rows
    are the documents and the columns are the
    topics. So entry in row X and column Y
    says how strongly document X belongs
    to topic Y.

    Second one is
    TopicTerm DataFrame where the rows
    are the topics and the columns are the
    terms. So entry in row X and column Y
    says how strongly term Y belongs
    to topic X.

    Examples
    --------
    Using Clustering:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, band, guitar"])
    >>> s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> s_cluster = s_tfidf.pipe(hero.normalize).pipe(hero.pca, n_components=2).pipe(hero.kmeans, n_clusters=2)
    >>> s_document_topic, s_topic_term = hero.topic_matrices(s_tfidf, s_cluster)
    >>> s_document_topic # doctest: +SKIP
    Document Topic Matrix   
                          0  1
    0                     1  0
    1                     0  1
    2                     1  0
    3                     0  1
    >>> s_topic_term # doctest: +SKIP
      Topic Term Matrix                                                                                
                   band  football       fun    guitar     music orchestra    soccer    sports    violin
    0          0.000000  3.021651  1.916291  0.000000  0.000000  0.000000  1.916291  3.021651  0.000000
    1          1.916291  0.000000  0.000000  1.916291  3.021651  1.916291  0.000000  0.000000  1.916291

    Using LDA:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, band, guitar"])
    >>> s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> s_lda = s_tfidf.pipe(hero.lda, n_components=2)
    >>> s_document_topic, s_topic_term = hero.topic_matrices(s_tfidf, s_lda)
    >>> s_document_topic # doctest: +SKIP
      Document Topic Matrix          
                      0         1
    0              0.912814  0.087186
    1              0.082094  0.917906
    2              0.912814  0.087186
    3              0.875660  0.124340
    >>> s_topic_term # doctest: +SKIP
      Topic Term Matrix                                                                                
                   band  football       fun    guitar     music orchestra    soccer    sports    violin
    0          1.678019  2.758205  1.749217  1.678019  1.447000  0.157316  1.749217  2.758205  0.157316
    1          0.238271  0.263446  0.167074  0.238271  1.574651  1.758974  0.167074  0.263446  1.758974

    See Also
    --------
    TODO add tutorial link

    """
    # Bool to note whether a clustering function or topic modelling
    # functions was used for s_document_topic.
    clustering_function_used = s_document_topic.dtype.name == "category"

    if not clustering_function_used:
        # Here, s_document_topic is output of hero.lda or hero.truncatedSVD.

        document_term_matrix = s_document_term.sparse.to_coo()
        document_topic_matrix = np.array(list(s_document_topic))
        n_topics = len(document_topic_matrix[0])

    else:
        # Here, s_document_topic is output of some hero clustering function.

        # First remove documents that are not assigned to any cluster.
        # They have clusterID ==  -1.
        indexes_of_unassigned_documents = s_document_topic == -1
        s_document_term = s_document_term[~indexes_of_unassigned_documents]
        s_document_topic = s_document_topic[~indexes_of_unassigned_documents]
        s_document_topic = s_document_topic.cat.remove_unused_categories()

        document_term_matrix = s_document_term.sparse.to_coo()

        # Construct document_topic_matrix from the cluster category Series
        # as described in the docstring.
        n_rows = len(s_document_topic.index)  # n_rows = number of documents
        # n_cols = number of clusters
        n_topics = n_cols = len(s_document_topic.values.categories)

        # Will get binary matrix:
        # document_topic_matrix[X][Y] = 1 <=> document X is in cluster Y.
        # We construct this matrix sparsely in CSR format
        # -> need the data (will only insert 1s, nothing else),
        # the rows (so in which rows we want to insert, which is all of them
        # as every document belongs to a cluster),
        # and we need the columns (so in which cluster we want to insert,
        # which is exactly the clusterID values).
        data = [1 for _ in range(n_rows)]  # Will insert one 1 per row.
        rows = range(n_rows)  # rows are just [0, 1, ..., n_rows]
        columns = s_document_topic.values

        # Construct the sparse matrix.
        document_topic_matrix = csr_matrix(
            (data, (rows, columns)), shape=(n_rows, n_cols)
        )

    topic_term_matrix = document_topic_matrix.T * document_term_matrix

    # Create s_document_topic and s_topic_term (both multiindexed)

    # Create s_document_topic
    s_document_topic_columns = pd.MultiIndex.from_product(
        [["Document Topic Matrix"], range(n_topics)]
    )

    if issparse(document_topic_matrix):
        s_document_topic = pd.DataFrame.sparse.from_spmatrix(
            document_topic_matrix,
            columns=s_document_topic_columns,
            index=s_document_term.index,
        )

    else:
        s_document_topic = pd.DataFrame(
            document_topic_matrix,
            columns=s_document_topic_columns,
            index=s_document_term.index,
            dtype="Sparse",
        )

    # Create s_topic_term
    s_topic_term_columns = pd.MultiIndex.from_product(
        [["Topic Term Matrix"], s_document_term.columns.levels[1].tolist()]
    )

    if issparse(topic_term_matrix):
        s_topic_term = pd.DataFrame.sparse.from_spmatrix(
            topic_term_matrix, columns=s_topic_term_columns
        )

    else:
        s_topic_term = pd.DataFrame(
            topic_term_matrix, columns=s_topic_term_columns, dtype="Sparse"
        )

    return s_document_topic, s_topic_term


def relevant_words_per_topic(
    s_document_term,
    s_document_topic_distribution,
    s_topic_term_distribution,
    n_words=10,
    return_figure=False,
):
    """
    Use `LDAvis <https://github.com/bmabey/pyLDAvis>`_ 
    to find the most relevant words for each topic.

    First input is a DocumentTerm Matrix, so the
    output of one of
    - :meth:`texthero.representation.tfidf`
    - :meth:`texthero.representation.count`
    - :meth:`texthero.representation.term_frequency`.

    The document-term-matrix has to include all
    terms that are present in the documents
    (i.e. you _cannot_ use the parameters max_df,
    min_df, or max_features).

    Second input is a DocumentTopic Distribution,
    so the l1-normalized (e.g. with :meth:`hero.representation.normalize`_)
    first output of :meth:`hero.visualization.topic_matrices`_.

    Third input is a TopicTerm Distribution,
    so the l1-normalized (e.g. with :meth:`hero.representation.normalize`_)
    second output of :meth:`hero.visualization.topic_matrices`_.

    This function uses the three given relations
    (documents->terms, documents->topics, topics->terms)
    to find and return the most relevant words for each topic.
    The `pyLDAvis library <https://github.com/bmabey/pyLDAvis>`_
    is used to find relevant words.

    Parameters
    ----------
    s_document_term : pd.DataFrame
        Output of one of
        :meth:`texthero.representation.tfidf`,
        :meth:`texthero.representation.count`,
        :meth:`texthero.representation.term_frequency`.
        All terms from the corpus have to be present
        (i.e. you _cannot_ use the parameters max_df,
        min_df, or max_features when computing
        s_document_term).

    s_document_topic_distribution : pd.DataFrame
        L1-Normalized first output of
        :meth:`texthero.visualization.topic_matrices`.

    s_topic_term_distribution : pd.DataFrame
        L1-Normalized second output of
        :meth:`texthero.visualization.topic_matrices`.

    n_words: int, default to 5
        Number of top words per topic, should
        be <= 30.

    Returns
    -------
    Pandas Series with the topic IDs as index and
    a list of n_words relevant words per
    topic as values.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, band, guitar"])
    >>> s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> s_cluster = s_tfidf.pipe(hero.normalize).pipe(hero.pca, n_components=2).pipe(hero.kmeans, n_clusters=2)
    >>> s_document_topic, s_topic_term = hero.topic_matrices(s_tfidf, s_cluster)
    >>> s_document_topic_distribution = hero.normalize(s_document_topic, norm="l1")
    >>> s_topic_term_distribution = hero.normalize(s_topic_term, norm="l1")
    >>> hero.relevant_words_per_topic(s_tfidf, s_document_topic_distribution, s_topic_term_distribution, n_words=2) # doctest: +SKIP
    Topic
    0       [music, violin]
    1    [sports, football]
    dtype: object

    See Also
    --------
    `pyLDAvis <https://pyldavis.readthedocs.io/en/latest/>`_
    for the methodology on how to find relevant terms.

    TODO add tutorial link
    """

    # Define parameters for pyLDAvis.
    vocab = s_document_term.columns.levels[1].tolist()
    doc_lengths = list(s_document_term.sum(axis=1))
    term_frequency = list(s_document_term.sum(axis=0))

    doc_topic_dists = s_document_topic_distribution.values.tolist()
    topic_term_dists = s_topic_term_distribution.values.tolist()

    # Create pyLDAvis visualization.
    figure = pyLDAvis.prepare(
        **{
            "vocab": vocab,
            "doc_lengths": doc_lengths,
            "term_frequency": term_frequency,
            "doc_topic_dists": doc_topic_dists,
            "topic_term_dists": topic_term_dists,
            "R": 15,
            "sort_topics": False,
        }
    )

    if return_figure:
        return figure

    # Extract relevant data from LDAvis output.
    # Most of the output is only useful for
    # the visualization internally (e.g.
    # term frequencies, ...).
    # We're only interested in the
    # relevant words per topic
    # which LDAvis returns in the "tinfo" field.

    pyLDAvis_data = figure.to_dict()

    # The top words per topic are in "tinfo".
    # We're not calculating/... anything below here,
    # only parsing the LDAvis output into a nice Series
    # we can return.
    df_topics_and_their_relevant_words = pd.DataFrame(pyLDAvis_data["tinfo"])

    # Throw out topic "Default"
    df_topics_and_their_relevant_words = df_topics_and_their_relevant_words[
        df_topics_and_their_relevant_words["Category"] != "Default"
    ]

    # Our topics / clusters begin at 0 -> use i-1 and rename e.g. "Topic4" to "3".
    n_topics = df_topics_and_their_relevant_words["Category"].nunique()

    replace_dict = {"Topic{}".format(i): i - 1 for i in range(1, n_topics + 1)}

    df_topics_and_their_relevant_words["Category"] = df_topics_and_their_relevant_words[
        "Category"
    ].replace(replace_dict)

    # Sort first by topic, then by word frequency.
    df_topics_and_their_relevant_words = df_topics_and_their_relevant_words.sort_values(
        ["Category", "Freq"], ascending=[1, 0]
    )

    # Group by topic and combine the relevant words into a list.
    s_topics_with_relevant_words = df_topics_and_their_relevant_words.groupby(
        "Category"
    )["Term"].apply(list)

    # Take the top n_words words for each topic.
    s_topics_with_relevant_words = s_topics_with_relevant_words.apply(
        lambda x: x[:n_words]
    )

    # Replace pyLDAvis names with ours.
    s_topics_with_relevant_words = s_topics_with_relevant_words.rename(None)
    s_topics_with_relevant_words.index.name = "Topic"

    return s_topics_with_relevant_words


def relevant_words_per_document(s_document_term, n_words=10):
    """
    Combine several Texthero functions to get the
    most relevant words of every document in your dataset.

    Using this function is equivalent to doing the following:

    ```python

    >>> # New Series where every document is its own cluster.
    >>> s_cluster = pd.Series(
    ...    np.arange(len(s_document_term)), index=s_document_term.index, dtype="category")  # doctest: +SKIP
    >>> s_document_topic, s_topic_term = hero.topic_matrices(s_document_term, s_cluster)  # doctest: +SKIP
    >>> s_document_topic_distribution = hero.normalize(s_document_topic, norm="l1")  # doctest: +SKIP
    >>> s_topic_term_distribution = hero.normalize(s_topic_term, norm="l1")  # doctest: +SKIP
    >>> relevant_words_per_topic(
    ...  s_document_term,
    ...  s_document_topic_distribution,
    ...  s_topic_term_distribution)  # doctest: +SKIP

    ```

    First input has to be output of one of 
    - :meth:`texthero.representation.tfidf`
    - :meth:`texthero.representation.count`
    - :meth:`texthero.representation.term_frequency`.

    The document-term-matrix has to include all
    terms that are present in the documents
    (i.e. you _cannot_ use the parameters max_df,
    min_df, or max_features).

    The function assigns every document
    to its own cluster (or "topic") and then uses
    :meth:`topic_matrices`_ and
    :meth:`relevant_words_per_topic`_ to find
    the most relevant words for every document
    with `pyLDAvis <https://pyldavis.readthedocs.io/en/latest/>`_ .

    Parameters
    ----------
    s_document_term: pd.DataFrame
        Output of one of
        :meth:`texthero.representation.tfidf`
        :meth:`texthero.representation.count`
        :meth:`texthero.representation.term_frequency`.
        All terms from the corpus have to be present
        (i.e. you _cannot_ use the parameters max_df,
        min_df, or max_features when computing
        s_document_term).

    n_words: int, default to 10
        Number of words to fetch per topic, should
        be <= 30.

    Returns
    -------
    Series with the documents as index and
    a list of n_words relevant words per
    document as values.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(
    ...    ["Football, Sports, Soccer, Golf",
    ...    "music, violin, orchestra",
    ...    "football, fun, sports",
    ...    "music, band, guitar"])
    >>> s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> hero.relevant_words_per_document(s_tfidf, n_words=2) # doctest: +SKIP
    0         [soccer, golf]
    1    [violin, orchestra]
    2          [fun, sports]
    3         [guitar, band]
    dtype: object
    >>> # We can see that the function tries to
    >>> # find terms that distinguish the documents,
    >>> # so e.g. "music" is not chosen for documents
    >>> # 1 and 3 as it's found in both of them.

    See Also
    --------
    `pyLDAvis <https://pyldavis.readthedocs.io/en/latest/>`_
    for the methodology on how to find relevant terms.

    :meth:`texthero.representation.topic_matrices`_

    :meth:`texthero.representation.relevant_words_per_topic`_

    TODO add tutorial link
    """

    # Create a categorical Series that has
    # one new cluster for every document.
    s_cluster = pd.Series(
        np.arange(len(s_document_term)), index=s_document_term.index, dtype="category"
    )

    # Get topic matrices.
    s_document_topic, s_topic_term = topic_matrices(s_document_term, s_cluster)

    # Get topic distributions through normalization.
    s_document_topic_distribution = normalize(s_document_topic, norm="l1")
    s_topic_term_distribution = normalize(s_topic_term, norm="l1")

    # Call relevant_words_per_topic with the new cluster series
    # (so every document is treated as one distinct "topic")
    s_relevant_words_per_document = relevant_words_per_topic(
        s_document_term,
        s_document_topic_distribution,
        s_topic_term_distribution,
        n_words=n_words,
    )

    return s_relevant_words_per_document.reindex(s_document_term.index)


"""
Normalization.
"""


def normalize(s: Union[pd.DataFrame, pd.Series], norm="l2") -> pd.Series:
    """
    Normalize every cell in a Pandas Series.

    Input can be VectorSeries or DocumentTermDF. For DocumentTermDFs,
    the sparseness is kept.

    Parameters
    ----------
    s: Pandas Series (VectorSeries) or MultiIndex Sparse DataFrame (DocumentTermDF)

    norm: str, default to "l2"
        One of "l1", "l2", or "max". The norm that is used.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> col = pd.MultiIndex.from_tuples([(0, "a"), (0, "b"), (1, "c"), (1, "d")])
    >>> s = pd.DataFrame([[1, 2, 3, 4],[4, 2, 7, 5],[2, 2, 3, 5],[1, 2, 9, 8]], columns=col).astype("Sparse")
    >>> hero.normalize(s, norm="max") # doctest: +SKIP
              0               1          
              a         b     c         d
    0  0.250000  0.500000  0.75  1.000000
    1  0.571429  0.285714  1.00  0.714286
    2  0.400000  0.400000  0.60  1.000000
    3  0.111111  0.222222  1.00  0.888889


    See Also
    --------
    Representation Series link TODO add link to tutorial

    `Norm on Wikipedia <https://en.wikipedia.org/wiki/Norm_(mathematics)>`_

    """
    isDocumentTermDF = _check_is_valid_DocumentTermDF(s)

    if isDocumentTermDF:
        s_coo = s.sparse.to_coo()
        s_for_vectorization = s_coo.astype("float64")
    else:
        s_for_vectorization = list(s)

    result = sklearn_normalize(
        s_for_vectorization, norm=norm
    )  # Can handle sparse input.

    if isDocumentTermDF:
        return pd.DataFrame.sparse.from_spmatrix(result, s.index, s.columns)
    else:
        return pd.Series(list(result), index=s.index)
