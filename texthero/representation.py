"""
Map words into vectors using different algorithms such as 
TF-IDF, word2vec or GloVe.
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sklearn_normalize
from scipy.sparse import coo_matrix

from typing import Optional, Union, Any
from texthero._types import (
    TextSeries,
    TokenSeries,
    VectorSeries,
    DataFrame,
    InputSeries,
)

from texthero import preprocessing

import logging
import warnings

# from texthero import pandas_ as pd_

"""
Helper
"""


# Warning message for not-tokenized inputs
_not_tokenized_warning_message = (
    "It seems like the given Pandas Series s is not tokenized. This"
    " function will tokenize it automatically using hero.tokenize(s)"
    " first. You should consider tokenizing it yourself first with"
    " hero.tokenize(s) in the future."
)


"""
Vectorization
"""


@InputSeries([TokenSeries, TextSeries])
def count(
    s: Union[TokenSeries, TextSeries],
    max_features: Optional[int] = None,
    min_df=1,
    max_df=1.0,
    binary=False,
) -> pd.DataFrame:
    """
    Represent a text-based Pandas Series using count.

    Rows of the returned DataFrame represent documents whereas 
    columns are terms. The value in the cell document-term is
    the number of the term in this document. The output is sparse.
    TODO add tutorial link

    The input Series should already be tokenized. If not, it will
    be tokenized before count is calculated.

    Parameters
    ----------
    s : Pandas Series (tokenized)

    max_features : int, optional, default=None
        Maximum number of features to keep. Will keep all features if set to
        None.

    min_df : float in range [0.0, 1.0] or int, optional, default=1
        When building the vocabulary ignore terms that have a document
        frequency (number of documents they appear in) strictly 
        lower than the given threshold.
        If float, the parameter represents a proportion of documents,
        integer absolute counts.

    max_df : float in range [0.0, 1.0] or int, optional, default=1.0
        Ignore terms that have a document frequency (number of documents they
        appear in) frequency strictly higher than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    binary : bool, optional, default=False
        If True, all non zero counts are set to 1.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"]).pipe(hero.tokenize)
    >>> hero.count(s) # doctest: +SKIP
       Sentence  one  two
    0         1    1    0
    1         1    0    1
   
    See Also
    --------

    TODO add tutorial link
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

    return pd.DataFrame.sparse.from_spmatrix(
        tf_vectors_csr, s.index, tf.get_feature_names()
    )


def term_frequency(
    s: pd.Series, max_features: Optional[int] = None, min_df=1, max_df=1.0,
) -> pd.DataFrame:
    """
    Represent a text-based Pandas Series using Term Frequency.

    Rows of the returned DataFrame represent documents whereas columns are
    terms. The value in the cell document-term is the frequency of the
    term in this document. The output is sparse.
    TODO add tutorial link

    The input Series should already be tokenized. If not, it will
    be tokenized before term_frequency is calculated.

    Parameters
    ----------
    s : Pandas Series (tokenized)

    max_features : int, optional, default=None
        Maximum number of features to keep. Will keep all features if set to
        None.

    min_df : float in range [0.0, 1.0] or int, optional, default=1
        When building the vocabulary ignore terms that have a document
        frequency (number of documents they appear in) strictly 
        lower than the given threshold.
        If float, the parameter represents a proportion of documents,
        integer absolute counts.

    max_df : float in range [0.0, 1.0] or int, optional, default=1.0
        Ignore terms that have a document frequency (number of documents they
        appear in) frequency strictly higher than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Text Text of doc one", "Text of of doc two", "Aha hi bnd one"]).pipe(hero.tokenize)
    >>> hero.term_frequency(s)  # doctest: +SKIP
    term_frequency                                      
                Aha Text   bnd  doc    hi   of   one  two
    0           0.00  0.4  0.00  0.2  0.00  0.2  0.20  0.0
    1           0.00  0.2  0.00  0.2  0.00  0.4  0.00  0.2
    2           0.25  0.0  0.25  0.0  0.25  0.0  0.25  0.0

    See Also
    --------
    TODO add tutorial link
    """
    # Term frequency is just the word counts for each document
    # with each document divided by the number of terms in the
    # document. That's just l1 normalization!
    s_term_frequency = s.pipe(
        count, max_features=max_features, min_df=min_df, max_df=max_df
    ).pipe(normalize, norm="l1")

    return s_term_frequency


def tfidf(s: pd.Series, max_features=None, min_df=1, max_df=1.0,) -> pd.DataFrame:
    """
    Represent a text-based Pandas Series using TF-IDF.

    Rows of the returned DataFrame represent documents whereas columns are
    terms. The value in the cell document-term is the tfidf-value of the
    term in this document. The output is sparse.

    *Term Frequency - Inverse Document Frequency (TF-IDF)* is a formula to
    calculate the _relative importance_ of the words in a document, taking
    into account the words' occurences in other documents. It consists of
    two parts:

    The *term frequency (tf)* tells us how frequently a term is present
    in a document, so tf(document d, term t) = number of times t appears
    in d.

    The *inverse document frequency (idf)* measures how _important_ or
    _characteristic_ a term is among the whole corpus (i.e. among all
    documents). Thus, idf(term t) = log((1 + number of documents) /
    (1 + number of documents where t is present)) + 1.

    Finally, tf-idf(document d, term t) = tf(d, t) * idf(t).

    Different from the `sklearn-implementation of tfidf
    <https://scikit-learn.org/stable/modules/generated/sklearn.feature_
    extraction.text.TfidfVectorizer.html>`, this function does *not* 
    normalize the output in any way, so the result is exactly what you 
    get applying the formula described above.

    The input Series should already be tokenized. If not, it will
    be tokenized before tfidf is calculated.

    Parameters
    ----------
    s : Pandas Series (tokenized)

    max_features : int, optional, default=None
        If not None, only the max_features most frequent tokens are used.

    min_df : float in range [0.0, 1.0] or int, optional, default=1
        When building the vocabulary ignore terms that have a document
        frequency (number of documents they appear in) strictly 
        lower than the given threshold.
        If float, the parameter represents a proportion of documents, 
        integer absolute counts.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        Ignore terms that have a document frequency (number of documents they
        appear in) frequency strictly higher than the given threshold.
        This arguments basically permits to remove corpus-specific stop 
        words. If float, the parameter represents a proportion of documents,
        integer absolute counts.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Hi Bye", "Test Bye Bye"]).pipe(hero.tokenize)
    >>> hero.tfidf(s) # doctest: +SKIP                    
        Bye        Hi      Test
    0   1.0  1.405465  0.000000
    1   2.0  0.000000  1.405465

    See Also
    --------
    `TF-IDF on Wikipedia <https://en.wikipedia.org/wiki/Tf-idf>`_

    TODO add tutorial link
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

    return pd.DataFrame.sparse.from_spmatrix(
        tfidf_vectors_csr, s.index, tfidf.get_feature_names()
    )


"""
Dimensionality reduction
"""


def pca(
    input_matrix: Union[pd.Series, pd.DataFrame], n_components=2, random_state=None
) -> pd.Series:
    """
    Perform principal component analysis on the given input.

    Principal Component Analysis (PCA) is a statistical method that is
    used to reveal where the variance in a dataset comes from. For 
    textual data, one could for example first represent a Series of 
    documents using :meth:`texthero.representation.tfidf` to get a vector
    representation of each document. Then, PCA can generate new vectors 
    from the tfidf representation that showcase the differences among
    the documents most strongly in fewer dimensions.

    For example, the tfidf vectors will have length 100 if hero.tfidf was
    called on a large corpus with max_features=100. Visualizing 100 
    dimensions is hard! Using PCA with n_components=3, every document will
    now get a vector of length 3, and the vectors will be chosen so that
    the document differences are easily visible. The corpus can now be 
    visualized in 3D and we can get a good first view of the data!

    In general, *pca* should be called after the text has already been
    represented to a matrix form.

    PCA cannot directly handle sparse input, so when calling pca on a
    sparse DataFrame, the input has to be expanded which can lead to
    memory problems with big datasets.

    Parameters
    ----------
    input_matrix : Pandas Series (VectorSeries) or DataFrame

    n_components : int or str, optional, default=2
        Number of components to keep (dimensionality of output vectors).
        If n_components is not set or None, all components are kept.
        If set to "mle", the number of components is
        automatically estimated.

    random_state : int, optional, default=None
        Pass an int for reproducible results across multiple function calls.


    Returns
    -------
    Pandas Series with the vector calculated by PCA for the document in
    every cell.

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
    `PCA on Wikipedia 
    <https://en.wikipedia.org/wiki/Principal_component_analysis>`_

    """
    # Default n_components=2 to enable users to easily plot the results.
    pca = PCA(n_components=n_components, random_state=random_state, copy=False)

    if isinstance(input_matrix, pd.DataFrame):
        values = input_matrix.values
    else:
        values = list(input_matrix)

    return pd.Series(list(pca.fit_transform(values)), index=input_matrix.index)


def nmf(
    input_matrix: Union[pd.Series, pd.DataFrame], n_components=2, random_state=None
) -> pd.Series:
    """
    Performs non-negative matrix factorization on the given input.

    Non-Negative Matrix Factorization (NMF) is often used in
    natural language processing to find clusters of similar
    texts (e.g. some texts in a corpus might be about sports
    and some about music, so they will differ in the usage
    of technical terms; see the example below). 

    Given a document-term matrix (so in
    texthero usually a Series after applying
    :meth:`texthero.representation.tfidf` or some other first 
    representation function that assigns a scalar (a weight) to each 
    word), NMF will find n_components many topics (clusters) and
    calculate a vector for each document that places it correctly among
    the topics.

    NMF can directly handle sparse input, so when calling nmf on a
    sparse DataFrame, the advantage of sparseness is kept.

    Parameters
    ----------
    input_matrix : Pandas Series (VectorSeries) or DataFrame

    n_components : int, optinal, default=2
        Number of components to keep (dimensionality of output vectors).
        If n_components is not set or None, all components are kept.

    random_state : int, optional, default=None
        Pass an int for reproducible results across multiple function calls.

    Returns
    -------
    Pandas Series with the vector calculated by NMF for the document in 
    every cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", 
    ...               "Music, Violin, Orchestra", "Football, Music"])
    >>> s = s.pipe(hero.clean).pipe(hero.tokenize).pipe(
    ...                                         hero.term_frequency
    ...                                                 )
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
    # Default n_components=2 to enable users to easily plot the results.
    nmf = NMF(n_components=n_components, init="random", random_state=random_state,)

    if isinstance(input_matrix, pd.DataFrame):
        input_matrix_coo = input_matrix.sparse.to_coo()
        input_matrix_for_vectorization = input_matrix_coo.astype("float64")
    else:
        input_matrix_for_vectorization = list(input_matrix)

    return pd.Series(
        list(nmf.fit_transform(input_matrix_for_vectorization)),
        index=input_matrix.index,
    )


def tsne(
    input_matrix: Union[pd.Series, pd.DataFrame],
    n_components=2,
    perplexity=30.0,
    learning_rate=200.0,
    n_iter=1000,
    random_state=None,
    n_jobs=-1,
) -> VectorSeries:
    """
    Performs t-Distributed Stochastic Neighbor Embedding on the given
    input.

    t-distributed Stochastic Neighbor Embedding (t-SNE) is
    a machine learning algorithm used to visualize high-dimensional data
    in fewer dimensions. In natural language processing, the
    high-dimensional data is usually a document-term matrix (so in 
    texthero usually a Series after applying
    :meth:`texthero.representation.tfidf` or some other first
    representation function that assigns a scalar (a weight) to each word)
    that is hard to visualize as there might be many terms. With t-SNE,
    every document gets a new, low-dimensional (n_components entries)
    vector in such a way that the differences / similarities between
    documents are preserved.

    T-SNE can directly handle sparse input, so when calling tsne on a
    sparse DataFrame, the advantage of sparseness is kept.

    Parameters
    ----------
    input_matrix : Pandas Series (VectorSeries) or DataFrame

    n_components : int, optional, default=2
        Number of components to keep (dimensionality of output vectors).
        If n_components is not set or None, all components are kept.

    perplexity : float, optional, default=30
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significanlty
        different results.

    learning_rate : float, optional, default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.

    n_iter : int, optional, default=1000
        Maximum number of iterations for the optimization. Should be at
        least 250.

    random_state : int, optional, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls.

    n_jobs : int, optional, default=-1
        The number of parallel jobs to run for neighbors search.
        ``-1`` means using all processors.

    Returns
    -------
    Pandas Series with the vector calculated by t-SNE for the document in
    every cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer",
    ...              "Music, Violin, Orchestra",  "Football, Music"])
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
    # Default n_components=2 to enable users to easily plot the results.
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    if isinstance(input_matrix, pd.DataFrame):
        input_matrix_coo = input_matrix.sparse.to_coo()
        input_matrix_for_vectorization = input_matrix_coo.astype("float64")
    else:
        input_matrix_for_vectorization = list(input_matrix)

    return pd.Series(
        list(tsne.fit_transform(input_matrix_for_vectorization)),
        index=input_matrix.index,
    )


"""
Clustering
"""


@InputSeries([VectorSeries, DataFrame])
def kmeans(
    input_matrix: Union[pd.Series, pd.DataFrame],
    n_clusters=5,
    n_init=10,
    max_iter=300,
    random_state=None,
    algorithm="auto",
) -> VectorSeries:
    """
    Performs K-means clustering algorithm on the given input.

    K-means clustering is used in natural language processing
    to separate texts into k clusters (groups) 
    (e.g. some texts in a corpus might be about sports
    and some about music, so they will differ in the usage
    of technical terms; the K-means algorithm uses this
    to separate them into two clusters). 

    Given a document-term matrix (so in
    texthero usually a Series after applying
    :meth:`texthero.representation.tfidf` or some other first 
    representation function that assigns a scalar (a weight) to each
    word), K-means will find k topics (clusters) and assign a topic to 
    each document.

    Kmeans can directly handle sparse input, so when calling kmeans on a
    sparse DataFrame, the advantage of sparseness is kept.

    Parameters
    ----------
    input_matrix: Pandas Series (VectorSeries) or DataFrame

    n_clusters: int, optional, default=5
        The number of clusters to separate the data into.

    n_init : int, optional, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, optional, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    random_state : int, optional, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    algorithm : {"auto", "full", "elkan"}, optional, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient on data with well-defined
        clusters, by using the triangle inequality. However it's more memory
        intensive.

    Returns
    -------
    Pandas Series with the cluster the document was assigned to in each 
    cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", 
    ...                 "music, violin, orchestra",
    ...                "football, fun, sports", "music, fun, guitar"])
    >>> s = s.pipe(hero.clean).pipe(hero.tokenize).pipe(
    ...                                         hero.term_frequency
    ...                                             )
    >>> hero.kmeans(s, n_clusters=2, random_state=42) # doctest: +SKIP
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
    `kmeans on Wikipedia 
    <https://en.wikipedia.org/wiki/K-means_clustering>`_

    """

    if isinstance(input_matrix, pd.DataFrame):
        input_matrix_coo = input_matrix.sparse.to_coo()
        input_matrix_for_vectorization = input_matrix_coo.astype("float64")
    else:
        input_matrix_for_vectorization = list(input_matrix)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        copy_x=True,
        algorithm=algorithm,
    ).fit(input_matrix_for_vectorization)
    return pd.Series(
        kmeans.predict(input_matrix_for_vectorization), index=input_matrix.index
    ).astype("category")


@InputSeries([VectorSeries, DataFrame])
def dbscan(
    input_matrix: Union[pd.Series, pd.DataFrame],
    eps=0.5,
    min_samples=5,
    metric="euclidean",
    metric_params=None,
    leaf_size=30,
    n_jobs=-1,
) -> VectorSeries:
    """
    Perform DBSCAN clustering on the given input.

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
    :meth:`texthero.representation.tfidf` or some other first
    representation function that assigns a scalar (a weight) to each 
    word), DBSCAN will find topics (clusters) and assign a topic to 
    each document.

    DBSCAN can directly handle sparse input, so when calling dbscan on a
    sparse DataFrame, the advantage of sparseness is kept.

    Parameters
    ----------
    input_matrix: Pandas Series (VectorSeries) or DataFrame

    eps : float, optional, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data 
        set and distance function.

    min_samples : int, optional, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    metric : string or callable, optional, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. Use 
        `sorted(sklearn.neighbors.VALID_METRICS['brute'])`
        to see valid options.

    metric_params : dict, optional, default=None
        Additional keyword arguments for the metric function.

    leaf_size : int, optional, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    n_jobs : int, optional, default=-1
        The number of parallel jobs to run.
        ``-1`` means using all processors.

    Returns
    -------
    Pandas Series with the cluster the document was assigned to in each
    cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", 
    ...                "music, violin, orchestra", 
    ...                "football, fun, sports", "music, enjoy, guitar"])
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

    if isinstance(input_matrix, pd.DataFrame):
        input_matrix_coo = input_matrix.sparse.to_coo()
        input_matrix_for_vectorization = input_matrix_coo.astype("float64")
    else:
        input_matrix_for_vectorization = list(input_matrix)

    return pd.Series(
        DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            leaf_size=leaf_size,
            n_jobs=n_jobs,
        ).fit_predict(input_matrix_for_vectorization),
        index=input_matrix.index,
    ).astype("category")


@InputSeries([VectorSeries, DataFrame])
def meanshift(
    input_matrix: Union[pd.Series, pd.DataFrame],
    bandwidth=None,
    bin_seeding=False,
    min_bin_freq=1,
    cluster_all=True,
    n_jobs=-1,
    max_iter=300,
) -> VectorSeries:
    """
    Perform mean shift clustering on the given input.

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
    :meth:`texthero.representation.tfidf` or some other first
    representation function that assigns a scalar (a weight) to each
    word), mean shift will find topics (clusters) and assign a topic
    to each document.

    Menashift cannot directly handle sparse input, so when calling
    meanshift on a sparse DataFrame, the input has to be expanded
    which can lead to memory problems with big datasets.

    Parameters
    ----------
    input_matrix: Pandas Series (VectorSeries) or DataFrame

    bandwidth : float, optional, default=None
        Bandwidth used in the RBF kernel.

        If not given, the bandwidth is estimated.
        Estimating takes time at least quadratic in the number of samples
        (i.e. documents). For large datasets, it’s wise to set the 
        bandwidth to a small value.

    bin_seeding : bool, optional, default=False
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will 
        speed up the algorithm because fewer seeds will be initialized.

    min_bin_freq : int, optional, default=1
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds.

    cluster_all : bool, optional, default=True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    n_jobs : int, optional, default=-1
        The number of jobs to use for the computation.
        ``-1`` means using all processors

    max_iter : int, optional, default=300
        Maximum number of iterations, per seed point before the clustering
        operation terminates (for that seed point), if has not converged
        yet.

    Returns
    -------
    Pandas Series with the cluster the document was assigned to in each
    cell.

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

    if isinstance(input_matrix, pd.DataFrame):
        vectors = input_matrix.values
    else:
        vectors = list(input_matrix)

    return pd.Series(
        MeanShift(
            bandwidth=bandwidth,
            bin_seeding=bin_seeding,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            n_jobs=n_jobs,
            max_iter=max_iter,
        ).fit_predict(vectors),
        index=input_matrix.index,
    ).astype("category")


"""
Topic modelling
"""

# TODO.

"""
Normalization.
"""


def normalize(input_matrix: Union[pd.DataFrame, pd.Series], norm="l2") -> pd.Series:
    """
    Normalize every cell in a Pandas Series.

    Input can be VectorSeries or DataFrames. For sparse DataFrames,
    the sparseness is kept.

    Parameters
    ----------
    input_matrix: Pandas Series (VectorSeries) or DataFrame

    norm: str, optional, default="l2"
        One of "l1", "l2", or "max". The norm that is used.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> col = ["a","b","c", "d"]
    >>> s = pd.DataFrame([[1, 2, 3, 4],[4, 2, 7, 5],[2, 2, 3, 5],[1, 2, 9, 8]], 
    ...                   columns=col).astype("Sparse")
    >>> hero.normalize(s, norm="max") # doctest: +SKIP      
              a         b     c         d
    0  0.250000  0.500000  0.75  1.000000
    1  0.571429  0.285714  1.00  0.714286
    2  0.400000  0.400000  0.60  1.000000
    3  0.111111  0.222222  1.00  0.888889


    See Also
    --------
    DataFrame link TODO add link to tutorial

    `Norm on Wikipedia
    <https://en.wikipedia.org/wiki/Norm_(mathematics)>`_

    """
    isDataFrame = isinstance(input_matrix, pd.DataFrame)

    if isDataFrame:
        input_matrix_coo = input_matrix.sparse.to_coo()
        input_matrix_for_vectorization = input_matrix_coo.astype("float64")
    else:
        input_matrix_for_vectorization = list(input_matrix)

    result = sklearn_normalize(
        input_matrix_for_vectorization, norm=norm
    )  # Can handle sparse input.

    if isDataFrame:
        return pd.DataFrame.sparse.from_spmatrix(
            result, input_matrix.index, input_matrix.columns
        )
    else:
        return pd.Series(list(result), index=input_matrix.index)
