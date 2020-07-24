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

from texthero import preprocessing

import logging
import warnings

# from texthero import pandas_ as pd_

"""
Helper
"""


def representation_series_to_flat_series(
    s: Union[pd.Series, pd.Series.sparse],
    index: pd.Index = None,
    fill_missing_with: Any = np.nan,
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

    fill_missing_with : Any, default to np.nan
        Value to fill the NaNs (missing values) with. This _does not_ mean
        that existing values that are np.nan are replaced, but rather that
        features that are not present in one document but present in others
        are filled with fill_missing_with. See example below.


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
    >>> hero.representation_series_to_flat_series(s, fill_missing_with=0.0)
    document
    doc0    [3.0, 0.0, nan]
    doc1    [0.0, 4.0, 0.0]
    dtype: object

    """
    s = s.unstack(fill_value=fill_missing_with)

    if index is not None:
        s = s.reindex(index, fill_value=fill_missing_with)
        # Reindexing makes the documents for which no values
        # are present in the Sparse Representation Series
        # "reappear" correctly.

    s = pd.Series(s.values.tolist(), index=s.index)

    s.rename_axis("document", inplace=True)

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


def count(
    s: pd.Series,
    max_features: Optional[int] = None,
    min_df=1,
    max_df=1.0,
    return_feature_names=False,
) -> pd.Series:
    """
    Represent a text-based Pandas Series using count.

    The input Series should already be tokenized. If not, it will
    be tokenized before count is calculated.

    Parameters
    ----------
    s : Pandas Series (tokenized)

    max_features : int, optional, default to None.
        Maximum number of features to keep. Will keep all features if set to None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency (number of documents they appear in) strictly 
        lower than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        Ignore terms that have a document frequency (number of documents they appear in)
        frequency strictly higher than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    return_features_names : Boolean, False by Default
        If True, return a tuple (*count_series*, *features_names*)


    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"]).pipe(hero.tokenize)
    >>> hero.count(s)
    0    [1, 1, 0]
    1    [1, 0, 1]
    dtype: object
    
    To return the features_names:
    
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"]).pipe(hero.tokenize)
    >>> hero.count(s, return_feature_names=True)
    (0    [1, 1, 0]
    1    [1, 0, 1]
    dtype: object, ['Sentence', 'one', 'two'])

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
    )
    s = pd.Series(tf.fit_transform(s).toarray().tolist(), index=s.index)

    if return_feature_names:
        return (s, tf.get_feature_names())
    else:
        return s


def term_frequency(
    s: pd.Series,
    max_features: Optional[int] = None,
    min_df=1,
    max_df=1.0,
    return_feature_names=False,
) -> pd.Series:

    """
    Represent a text-based Pandas Series using term frequency.

    The input Series should already be tokenized. If not, it will
    be tokenized before term_frequency is calculated.

    Parameters
    ----------
    s : Pandas Series (tokenized)

    max_features : int, optional, default to None.
        Maximum number of features to keep. Will keep all features if set to None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency (number of documents they appear in) strictly 
        lower than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        Ignore terms that have a document frequency (number of documents they appear in)
        frequency strictly higher than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    return_features_names : Boolean, False by Default
        If True, return a tuple (*count_series*, *features_names*)


    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"]).pipe(hero.tokenize)
    >>> hero.term_frequency(s)
    0    [0.25, 0.25, 0.0]
    1    [0.25, 0.0, 0.25]
    dtype: object
    
    To return the features_names:
    
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Sentence one", "Sentence two"]).pipe(hero.tokenize)
    >>> hero.term_frequency(s, return_feature_names=True)
    (0    [0.25, 0.25, 0.0]
    1    [0.25, 0.0, 0.25]
    dtype: object, ['Sentence', 'one', 'two'])

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

    cv_fit_transform = tf.fit_transform(s).toarray()
    total_count = np.sum(cv_fit_transform)
    s = pd.Series(np.divide(cv_fit_transform, total_count).tolist(), index=s.index)

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

    Different from the `sklearn-implementation of tfidf <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`,
    this function does *not* normalize the output in any way,
    so the result is exactly what you
    get applying the formula described above.

    The input Series should already be tokenized. If not, it will
    be tokenized before tfidf is calculated.

    If working with big pandas Series, you might want to limit
    the number of features through the max_features parameter.

    Parameters
    ----------
    s : Pandas Series (tokenized)

    max_features : int, optional, default to None.
        Maximum number of features to keep. Will keep all features if set to None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency (number of documents they appear in) strictly 
        lower than the given threshold.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        Ignore terms that have a document frequency (number of documents they appear in)
        frequency strictly higher than the given threshold.
        This arguments basically permits to remove corpus-specific stop words.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.

    return_features_names : Boolean, False by Default
        If True, return a tuple (*count_series*, *features_names*)


    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Hi Bye", "Test Bye Bye"]).pipe(hero.tokenize)
    >>> hero.tfidf(s, return_feature_names=True)
    (document
    0    [1.0, 1.4054651081081644, 0.0]
    1    [2.0, 0.0, 1.4054651081081644]
    dtype: object, ['Bye', 'Hi', 'Test'])

    See Also
    --------
    `TF-IDF on Wikipedia <https://en.wikipedia.org/wiki/Tf-idf>`_

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
    s_out = representation_series_to_flat_series(
        s_out, fill_missing_with=0.0, index=s.index
    )

    if return_feature_names:
        return s_out, feature_names
    else:
        return s_out


"""
Dimensionality reduction
"""


def pca(s, n_components=2) -> pd.Series:
    """
    Perform principal component analysis on the given Pandas Series.

    Principal Component Analysis (PCA) is a statistical method that is used
    to reveal where the variance in a dataset comes from. For textual data,
    one could for example first represent a Series of documents using
    :meth:`texthero.representation.tfidf` to get a vector representation
    of each document. Then, PCA can generate new vectors from the tfidf representation
    that showcase the differences among the documents most strongly in fewer dimensions.

    For example, the tfidf vectors will have length 100 if hero.tfidf was called
    on a large corpus with max_features=100. Visualizing 100 dimensions is hard!
    Using PCA with n_components=3, every document will now get a vector of
    length 3, and the vectors will be chosen so that the document differences
    are easily visible. The corpus can now be visualized in 3D and we can
    get a good first view of the data!

    In general, *pca* should be called after the text has already been represented to a matrix form.

    Parameters
    ----------
    s : Pandas Series

    n_components : Int. Default is 2.
        Number of components to keep (dimensionality of output vectors).
        If n_components is not set or None, all components are kept.

    random_state : int, RandomState instance, default=None
        Pass an int for reproducible results across multiple function calls.


    Returns
    -------
    Pandas Series with the vector calculated by PCA for the document in every cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football is great", "Hi, I'm Texthero, who are you? Tell me!"])
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

    :meth:`tfidf` to compute TF-IDF and :meth:`term_frequency` to compute term frequency
 
    """
    pca = PCA(n_components=n_components)
    return pd.Series(pca.fit_transform(list(s)).tolist(), index=s.index)


def nmf(s, n_components=2) -> pd.Series:
    """
    Performs non-negative matrix factorization.

    Non-Negative Matrix Factorization (NMF) is often used in
    natural language processing to find clusters of similar
    texts (e.g. some texts in a corpus might be about sports
    and some about music, so they will differ in the usage
    of technical terms; see the example below). 

    Given a document-term matrix (so in
    texthero usually a Series after applying :meth:`texthero.representation.tfidf`
    or some other first representation function that assigns a scalar (a weight)
    to each word), NMF will find n_components many topics (clusters)
    and calculate a vector for each document that places it
    correctly among the topics.


    Parameters
    ----------
    s : Pandas Series

    n_components : Int. Default is 2.
        Number of components to keep (dimensionality of output vectors).
        If n_components is not set or None, all components are kept.

    Returns
    -------
    Pandas Series with the vector calculated by NMF for the document in every cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "Music, Violin, Orchestra", "Football, Music"])
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
    `NMF on Wikipedia <https://en.wikipedia.org/wiki/Non-negative_matrix_factorization>`_

    :meth:`tfidf` to compute TF-IDF and :meth:`term_frequency` to compute term frequency

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
) -> pd.Series:
    """
    Performs TSNE on the given pandas series.

    t-distributed Stochastic Neighbor Embedding (t-SNE) is
    a machine learning algorithm used to visualize high-dimensional data in fewer
    dimensions. In natural language processing, the high-dimensional
    data is usually a document-term matrix
    (so in texthero usually a Series after applying :meth:`texthero.representation.tfidf`
    or some other first representation function that assigns a scalar (a weight)
    to each word) that is hard to visualize as there
    might be many terms. With t-SNE, every document
    gets a new, low-dimensional (n_components entries)
    vector in such a way that the differences / similarities between
    documents are preserved.


    Parameters
    ----------
    s : Pandas Series

    n_components : int, default is 2.
        Number of components to keep (dimensionality of output vectors).
        If n_components is not set or None, all components are kept.

    perplexity : float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significanlty
        different results.

    early_exaggeration : float, optional (default: 12.0)
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.

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

    n_iter_without_progress : int, optional (default: 300)
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.

    min_grad_norm : float, optional (default: 1e-7)
        If the gradient norm is below this threshold, the optimization will
        be stopped.

    metric : string or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter.

        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.

    init : string or numpy array, optional (default: "random")
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int, RandomState instance, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls. Note that different
        initializations might result in different local minima of the cost
        function.

    method : string (default: 'barnes_hut')
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.

    angle : float (default: 0.5)
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="precomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``-1`` means using all processors.

    Returns
    -------
    Pandas Series with the vector calculated by t-SNE for the document in every cell.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "Music, Violin, Orchestra", "Football, Music"])
    >>> s = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.term_frequency)
    >>> hero.tsne(s, random_state=42) # doctest: +SKIP
    0      [-18.833383560180664, -276.800537109375]
    1     [-210.60179138183594, 143.00535583496094]
    2    [-478.27984619140625, -232.97410583496094]
    dtype: object

    See also
    --------
    `t-SNE on Wikipedia <https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding>`_

    :meth:`tfidf` to compute TF-IDF and :meth:`term_frequency` to compute term frequency

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
    Performs K-means clustering algorithm.

    K-means clustering is used in natural language processing
    to separate texts into k clusters (groups) 
    (e.g. some texts in a corpus might be about sports
    and some about music, so they will differ in the usage
    of technical terms; the K-means algorithm uses this
    to separate them into two clusters). 

    Given a document-term matrix (so in
    texthero usually a Series after applying :meth:`texthero.representation.tfidf`
    or some other first representation function that assigns a scalar (a weight)
    to each word), K-means will find k topics (clusters)
    and assign a topic to each document.

    Parameters
    ----------
    s: Pandas Series

    n_clusters: Int, default to 5.
        The number of clusters to separate the data into.

    init : {'k-means++', 'random', ndarray, callable}, default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance, default=None
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

    :meth:`tfidf` to compute TF-IDF and :meth:`term_frequency` to compute term frequency

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
    return pd.Series(kmeans.predict(vectors), index=s.index).astype("category")


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

    Density-based spatial clustering of applications with noise (DBSCAN)
    is used in natural language processing
    to separate texts into clusters (groups)
    (e.g. some texts in a corpus might be about sports
    and some about music, so they will differ in the usage
    of technical terms; the DBSCAN algorithm uses this
    to separate them into clusters). It chooses the
    number of clusters on its own.

    Given a document-term matrix (so in
    texthero usually a Series after applying :meth:`texthero.representation.tfidf`
    or some other first representation function that assigns a scalar (a weight)
    to each word), DBSCAN will find topics (clusters)
    and assign a topic to each document.

    Parameters
    ----------
    s: Pandas Series

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
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

    p : float, default=None
        The power of the Minkowski metric to be used to calculate distance
        between points.

    n_jobs : int, default=None
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
    document
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

    :meth:`tfidf` to compute TF-IDF and :meth:`term_frequency` to compute term frequency

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
    ).astype("category")


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

    Mean shift clustering
    is used in natural language processing
    to separate texts into clusters (groups)
    (e.g. some texts in a corpus might be about sports
    and some about music, so they will differ in the usage
    of technical terms; the mean shift algorithm uses this
    to separate them into clusters). It chooses the
    number of clusters on its own.

    Given a document-term matrix (so in
    texthero usually a Series after applying :meth:`texthero.representation.tfidf`
    or some other first representation function that assigns a scalar (a weight)
    to each word), mean shift will find topics (clusters)
    and assign a topic to each document.

    Parameters
    ----------
    s: Pandas Series

    bandwidth : float, default=None
        Bandwidth used in the RBF kernel.

        If not given, the bandwidth is estimated using
        sklearn.cluster.estimate_bandwidth; see the documentation for that
        function for hints on scalability.

    seeds : array-like of shape (n_samples, n_features), default=None
        Seeds used to initialize kernels.

    bin_seeding : bool, default=False
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        The default value is False.
        Ignored if seeds argument is not None.

    min_bin_freq : int, default=1
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds.

    cluster_all : bool, default=True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    n_jobs : int, default=None
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

    :meth:`tfidf` to compute TF-IDF and :meth:`term_frequency` to compute term frequency

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
    ).astype("category")


"""
Topic modelling
"""

# TODO.
