"""
Visualize insights and statistics of a text-based Pandas DataFrame.
"""

import pandas as pd
import numpy as np
import plotly.express as px

from wordcloud import WordCloud

from texthero import preprocessing
from texthero._types import TextSeries, InputSeries
import string

from matplotlib.colors import LinearSegmentedColormap as lsg
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import normalize as sklearn_normalize

import pyLDAvis
from pyLDAvis import display as notebook_display
from pyLDAvis import show as browser_display

from collections import Counter
from typing import Tuple


def scatterplot(
    df: pd.DataFrame,
    col: str,
    color: str = None,
    hover_name: str = None,
    hover_data: [] = None,
    title="",
    return_figure=False,
):
    """
    Show scatterplot of DataFrame column using python plotly scatter.

    Plot the values in column col. For example, if every cell in df[col]
    is a list of three values (e.g. from doing PCA with 3 components),
    a 3D-Plot is created and every cell entry [x, y, z] is visualized
    as the point (x, y, z).

    Parameters
    ----------
    df: DataFrame with a column to be visualized.

    col: str
        The name of the column of the DataFrame to use for x and y (and z)
        axis.

    color: str, default to None.
        Name of the column to use for coloring (rows with same value get same
        color).

    hover_name: str, default to None
        Name of the column to supply title of hover data when hovering over a
        point.

    hover_data: List[str], default to [].
        List of column names to supply data when hovering over a point.

    title: str, default to "".
        Title of the plot.

    return_figure: optional, default to False.
        Function returns the figure instead of showing it if set to True.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> df = pd.DataFrame(["Football, Sports, Soccer",
    ...                    "music, violin, orchestra", "football, fun, sports",
    ...                    "music, fun, guitar"], columns=["texts"])
    >>> df["texts"] = hero.clean(df["texts"]).pipe(hero.tokenize)
    >>> df["pca"] = (
    ...             hero.tfidf(df["texts"])
    ...                 .pipe(hero.pca, n_components=3)
    ... ) # TODO: when others get Representation Support: remove flatten
    >>> df["topics"] = (
    ...                hero.tfidf(df["texts"])
    ...                    .pipe(hero.kmeans, n_clusters=2)
    ... ) # TODO: when others get Representation Support: remove flatten
    >>> hero.scatterplot(df, col="pca", color="topics",
    ...                  hover_data=["texts"]) # doctest: +SKIP
    """

    plot_values = np.stack(df[col], axis=1)
    dimension = len(plot_values)

    if dimension < 2 or dimension > 3:
        raise ValueError(
            "The column you want to visualize has dimension < 2 or dimension > 3."
            " The function can only visualize 2- and 3-dimensional data."
        )

    if dimension == 2:
        x, y = plot_values[0], plot_values[1]

        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color,
            hover_data=hover_data,
            title=title,
            hover_name=hover_name,
        )

    else:
        x, y, z = plot_values[0], plot_values[1], plot_values[2]

        fig = px.scatter_3d(
            df,
            x=x,
            y=y,
            z=z,
            color=color,
            hover_data=hover_data,
            title=title,
            hover_name=hover_name,
        )

    if return_figure:
        return fig
    else:
        fig.show()


"""
Wordcloud
"""


@InputSeries(TextSeries)
def wordcloud(
    s: TextSeries,
    font_path: str = None,
    width: int = 400,
    height: int = 200,
    max_words=200,
    mask=None,
    contour_width=0,
    contour_color="PAPAYAWHIP",
    min_font_size=4,
    background_color="PAPAYAWHIP",
    max_font_size=None,
    relative_scaling="auto",
    colormap=None,
    return_figure=False,
):
    """
    Plot wordcloud image using WordCloud from word_cloud package.

    Most of the arguments are very similar if not equal to the mother
    function. In constrast, all words are taken into account when computing
    the wordcloud, inclusive stopwords. They can be easily removed with
    preprocessing.remove_stopwords.

    Words are computed using generate_from_frequencies.

    To reduce blur in the wordcloud image, `width` and `height` should be at
    least 400.

    Parameters
    ----------
    s : :class:`texthero._types.TextSeries`

    font_path : str
        Font path to the font that will be used (OTF or TTF). Defaults to
        DroidSansMono path on a Linux machine. If you are on another OS or
        don't have this font, you need to adjust this path.

    width : int
        Width of the canvas.

    height : int
        Height of the canvas.

    max_words : number (default=200)
        The maximum number of words.

    mask : nd-array or None (default=None)
        When set, gives a binary mask on where to draw words. When set, width
        and height will be ignored and the shape of mask will be used instead.
        All white (#FF or #FFFFFF) entries will be considerd "masked out"
        while other entries will be free to draw on.

    contour_width: float (default=0)
        If mask is not None and contour_width > 0, draw the mask contour.

    contour_color: color value (default="PAPAYAWHIP")
        Mask contour color.

    min_font_size : int (default=4)
        Smallest font size to use. Will stop when there is no more room in
        this size.

    background_color : color value (default="PAPAYAWHIP")
        Background color for the word cloud image.

    max_font_size : int or None (default=None)
        Maximum font size for the largest word. If None, height of the image
        is used.

    relative_scaling : float (default='auto')
        Importance of relative word frequencies for font-size.  With
        relative_scaling=0, only word-ranks are considered.  With
        relative_scaling=1, a word that is twice as frequent will have twice
        the size.  If you want to consider the word frequencies and not only
        their rank, relative_scaling around .5 often looks good.
        If 'auto' it will be set to 0.5 unless repeat is true, in which
        case it will be set to 0.

    colormap : string or matplotlib colormap, default="viridis"
        Matplotlib colormap to randomly draw colors from for each word.

    """
    text = s.str.cat(sep=" ")

    if colormap is None:

        # Custom palette.
        # TODO move it under tools.
        corn = (255.0 / 256, 242.0 / 256, 117.0 / 256)
        mango_tango = (255.0 / 256, 140.0 / 256, 66.0 / 256)
        crayola = (63.0 / 256, 136.0 / 256, 197.0 / 256)
        crimson = (215.0 / 256, 38.0 / 256, 61.0 / 256)
        oxford_blue = (2.0 / 256, 24.0 / 256, 43.0 / 256)

        texthero_cm = lsg.from_list(
            "texthero", [corn, mango_tango, crayola, crimson, oxford_blue]
        )

        colormap = texthero_cm

    words = s.str.cat(sep=" ").split()

    wordcloud = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        max_words=max_words,
        mask=mask,
        contour_width=contour_width,
        contour_color=contour_color,
        min_font_size=min_font_size,
        background_color=background_color,
        max_font_size=max_font_size,
        relative_scaling=relative_scaling,
        colormap=colormap,
        # stopwords=[],  # TODO. Will use generate from frequencies.
        # normalize_plurals=False,  # TODO.
    ).generate_from_frequencies(dict(Counter(words)))

    # fig = px.imshow(wordcloud)
    # fig.show()

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    if return_figure:
        return fig


@InputSeries(TextSeries)
def top_words(s: TextSeries, normalize=False) -> pd.Series:
    r"""
    Return a pandas series with index the top words and as value the count.

    Tokenization: split by space and remove all punctuations that are not
    between characters.

    Parameters 
    ----------
    normalize : optional, default to False.
        When set to true, return normalized values.

    Examples
    --------
    >>> import pandas as pd
    >>> import texthero as hero
    >>> s = pd.Series("one two two three three three")
    >>> hero.top_words(s)
    three    3
    two      2
    one      1
    dtype: int64

    """

    # Replace all punctuation that are NOT in-between chacarters
    # This means, they have either a non word-bounding \B, are at the start ^, or at the end $
    # As re.sub replace all and not just the matching group, add matching parenthesis to the character
    # to keep during replacement.

    # TODO replace it with tokenizer.

    pattern = (
        rf"((\w)[{string.punctuation}](?:\B|$)|(?:^|\B)[{string.punctuation}](\w))"
    )

    return (
        s.str.replace(
            pattern, r"\2 \3"
        )  # \2 and \3 permits to keep the character around the punctuation.
        .str.split()  # now split by space
        .explode()  # one word for each line
        .value_counts(normalize=normalize)
    )


def _get_matrices_for_visualize_topics(
    s_document_term: pd.DataFrame,
    s_document_topic: pd.Series,
    clustering_function_used: bool,
):
    # TODO: add Hero types everywhere when they're merged
    """
    Helper function for visualize_topics. Used to extract and
    calculate the matrices that pyLDAvis needs.

    Recieves as first argument s_document_term, which is the output of
    tfidf / count / term_frequency. From this, s_document_term.values
    is the document_term_matrix in the code.

    Recieves as second argument s_document_topic, which is either
    the output of a clustering function (so a categorical Series)
    or the output of a topic modelling function (so a VectorSeries).

    In the first case (that's when clustering_function_used=True),
    we create the document_topic_matrix
    through the clusterIDs. So if document X is in cluster Y,
    then document_topic_matrix[X][Y] = 1.
    
    For example, when
    `s_document_topic = pd.Series([0, 2, 2, 1, 0, 1], dtype="category")`,
    then the document_topic_matrix is
    1 0 0
    0 0 1
    0 0 1
    0 1 0
    1 0 0
    0 1 0

    So e.g. document zero is in cluster 0, so document_topic_matrix[0][0] = 1.

    In the second case (that's when lda or truncatedSVD were used),
    their output is already the document_topic_matrix that relates
    documents to topics.

    We then have in both cases the document_term_matrix and the document_topic_matrix.
    pyLDAvis still needs the topic_term_matrix, which we get through
    topic_term_matrix = document_term_matrix.T * document_topic_matrix.

    """
    if not clustering_function_used:
        # Here, s_document_topic is output of hero.lda or hero.truncatedSVD.

        document_term_matrix = s_document_term.sparse.to_coo()
        document_topic_matrix = np.array(list(s_document_topic))

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
        n_cols = len(s_document_topic.values.categories)  # n_cols = number of clusters

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

    return s_document_term, s_document_topic, document_topic_matrix, topic_term_matrix


def _prepare_matrices_for_pyLDAvis(
    document_topic_matrix: np.matrix, topic_term_matrix: np.matrix
):
    # TODO: add types everywhere when they're merged
    """
    Helper function for visualize_topics. Used to prepare the
    document_topic_matrix and the topic_term_matrix for pyLDAvis.

    First normalizes both matrices to get the
    document_topic_distributions and topic_term_distributions matrix.
    For example, the first row of document_topic_distributions
    has the probabilities of document zero to belong to the
    different topics (so every row sums up to 1 (this is later
    checked by pyLDAvis)). 
    So document_topic_matrix[i][j] = proportion of document i
    that belongs to topic j.

    Then densify the (potentially) sparse matrices for pyLDAvis.
    """

    # Get distributions through normalization
    document_topic_distributions = sklearn_normalize(
        document_topic_matrix, norm="l1", axis=1
    )

    topic_term_distributions = sklearn_normalize(topic_term_matrix, norm="l1", axis=1)

    # Make sparse matrices dense for pyLDAvis

    if issparse(document_topic_distributions):
        document_topic_distributions = document_topic_distributions.toarray().tolist()
    else:
        document_topic_distributions = document_topic_distributions.tolist()

    if issparse(topic_term_distributions):
        topic_term_distributions = topic_term_distributions.toarray().tolist()
    else:
        topic_term_distributions = topic_term_distributions.tolist()

    return document_topic_distributions, topic_term_distributions


def visualize_topics(s_document_term: pd.DataFrame, s_document_topic: pd.Series):
    # TODO: add types everywhere when they're merged
    """
    Visualize the topics of your dataset. First input has
    to be output of one of 
    - :meth:`texthero.representation.tfidf`
    - :meth:`texthero.representation.count`
    - :meth:`texthero.representation.term_frequency`

    (tfidf suggested).

    Second input can either be the result of
    clustering, so output of one of
    - :meth:`texthero.representation.kmeans`
    - :meth:`texthero.representation.meanshift`
    - :meth:`texthero.representation.dbscan`

    or the result of :meth:`texthero.representation.lda`.

    The function uses the given clustering
    or topic modelling from the second input, which relates
    documents to topics. The first input
    relates documents to terms. From those
    two relations (documents->topics, documents->terms),
    the function calculates a distribution of
    documents to topics, and a distribution
    of topics to terms. These distributions
    are passed to `pyLDAvis <https://pyldavis.readthedocs.io/en/latest/>`_,
    which visualizes them.


    **To show the plot**:
    - Interactively in a Jupyter Notebook: use `hero.notebook_display(hero.visualize_topics(...))`
    - In a new browser window: `hero.browser_display(hero.visualize_topics(...))`

    Parameters
    ----------
    s_document_term: pd.DataFrame
        One of 
        :meth:`texthero.representation.tfidf`
        :meth:`texthero.representation.count`
        :meth:`texthero.representation.term_frequency`

    s_document_topic: pd.Series
        One of
        :meth:`texthero.representation.kmeans`
        :meth:`texthero.representation.meanshift`
        :meth:`texthero.representation.dbscan`
        (using clustering functkmeansions, documents
        that are not assigned to a cluster are
        not considered in the visualization)
        or one of
        :meth:`texthero.representation.lda`
        :meth:`texthero.representation.truncatedSVD`

    Examples
    --------
    Using Clustering:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, band, guitar"])
    >>> s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> s_cluster = s_tfidf.pipe(hero.normalize).pipe(hero.pca, n_components=2).pipe(hero.kmeans, n_clusters=2)
    >>> # Display in a new browser window:
    >>> hero.browser_display(hero.visualize_topics(s_tfidf, s_cluster)) # doctest: +SKIP
    >>> # Display inside the current Jupyter Notebook:
    >>> hero.notebook_display(hero.visualize_topics(s_tfidf, s_cluster)) # doctest: +SKIP

    Using LDA:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, band, guitar"])
    >>> s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> s_lda = s_tfidf.pipe(hero.lda, n_components=2)
    >>> # Display in a new browser window:
    >>> hero.browser_display(hero.visualize_topics(s_tfidf, s_lda)) # doctest: +SKIP
    >>> # Display inside the current Jupyter Notebook:
    >>> hero.notebook_display(hero.visualize_topics(s_tfidf, s_lda)) # doctest: +SKIP


    See Also
    --------
    `pyLDAvis <https://pyldavis.readthedocs.io/en/latest/>`_

    TODO add tutorial link

    """
    # Bool to note whether a clustering function or topic modelling
    # functions was used for s_document_topic.
    clustering_function_used = s_document_topic.dtype.name == "category"

    # Get / build matrices from input
    # (see helper function docstring for explanation)
    (
        s_document_term,
        s_document_topic,
        document_topic_matrix,
        topic_term_matrix,
    ) = _get_matrices_for_visualize_topics(
        s_document_term, s_document_topic, clustering_function_used
    )

    vocab = list(s_document_term.columns.levels[1])
    doc_lengths = list(s_document_term.sum(axis=1))
    term_frequency = list(s_document_term.sum(axis=0))

    # Prepare matrices for input to pyLDAvis
    # (see helper function docstring for explanation)
    (
        document_topic_distributions,
        topic_term_distributions,
    ) = _prepare_matrices_for_pyLDAvis(document_topic_matrix, topic_term_matrix)

    # Create pyLDAvis visualization.
    figure = pyLDAvis.prepare(
        **{
            "vocab": vocab,
            "doc_lengths": doc_lengths,
            "term_frequency": term_frequency,
            "doc_topic_dists": document_topic_distributions,
            "topic_term_dists": topic_term_distributions,
            "R": 15,
            "sort_topics": False,
        }
    )

    return figure


def top_words_per_topic(
    s_document_term: pd.DataFrame, s_clusters: pd.Series, n_words=5
):
    # TODO: add types everywhere when they're merged
    """
    Find the top words per topic of your dataset. First input has
    to be output of one of 
    - :meth:`texthero.representation.tfidf`
    - :meth:`texthero.representation.count`
    - :meth:`texthero.representation.term_frequency`

    (tfidf suggested).

    Second input has to be the result of
    clustering, so output of one of
    - :meth:`texthero.representation.kmeans`
    - :meth:`texthero.representation.meanshift`
    - :meth:`texthero.representation.dbscan`
    - :meth:`texthero.representation.topics_from_topic_model`

    The function uses the given clustering
    from the second input, which relates
    documents to topics. The first input
    relates documents to terms. From those
    two relations (documents->topics, documents->terms),
    the function calculates a distribution of
    documents to topics, and a distribution
    of topics to terms. These distributions
    are used to find the most relevant
    terms per topic.

    Parameters
    ----------
    s_document_term: pd.DataFrame
        One of 
        :meth:`texthero.representation.tfidf`
        :meth:`texthero.representation.count`
        :meth:`texthero.representation.term_frequency`

    s_clusters: pd.Series
        One of
        :meth:`texthero.representation.kmeans`
        :meth:`texthero.representation.meanshift`
        :meth:`texthero.representation.dbscan`
        :meth:`texthero.representation.topics_from_topic_model`

    n_words: int, default to 5
        Number of top words per topic, should
        be <= 30.

    Returns
    -------
    Series with the topic IDs as index and
    a list of n_words relevant words per
    topic as values.

    Examples
    --------
    Using Clustering:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, band, guitar"])
    >>> s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> s_cluster = s_tfidf.pipe(hero.normalize).pipe(hero.pca, n_components=2).pipe(hero.kmeans, n_clusters=2)
    >>> hero.top_words_per_topic(s_tfidf, s_cluster) # doctest: +SKIP
    0    [sports, football, soccer]
    1    [music, violin, orchestra]
    dtype: object

    See Also
    --------
    `pyLDAvis <https://pyldavis.readthedocs.io/en/latest/>`_
    and their methodology on how to find relevant terms.

    TODO add tutorial link

    """

    pyLDAvis_result = visualize_topics(s_document_term, s_clusters).to_dict()

    df_topics_and_their_top_words = pd.DataFrame(pyLDAvis_result["tinfo"])

    # Throw out topic "Default"
    df_topics_and_their_top_words = df_topics_and_their_top_words[
        df_topics_and_their_top_words["Category"] != "Default"
    ]

    n_topics = df_topics_and_their_top_words["Category"].nunique()

    # Our topics / clusters begin at 0 -> use i-1
    replace_dict = {"Topic{}".format(i): i - 1 for i in range(1, n_topics + 1)}

    df_topics_and_their_top_words["Category"] = df_topics_and_their_top_words[
        "Category"
    ].replace(replace_dict)

    df_topics_and_their_top_words = df_topics_and_their_top_words.sort_values(
        ["Category", "Freq"], ascending=[1, 0]
    )

    s_topics_with_top_words = df_topics_and_their_top_words.groupby("Category")[
        "Term"
    ].apply(list)

    s_topics_with_top_words = s_topics_with_top_words.apply(lambda x: x[:n_words])

    # Remove series name "Term" from pyLDAvis
    s_topics_with_top_words = s_topics_with_top_words.rename(None)

    return s_topics_with_top_words


def top_words_per_document(s_document_term: pd.DataFrame, n_words=3):
    # TODO: add types everywhere when they're merged
    """
    Find the top words per document of your dataset. First input has
    to be output of one of 
    - :meth:`texthero.representation.tfidf`
    - :meth:`texthero.representation.count`
    - :meth:`texthero.representation.term_frequency`

    (tfidf suggested).

    The function assigns every document
    to its own cluster (or "topic") and then uses
    :meth:`top_words_per_topic` to find
    the top words for every document.

    Parameters
    ----------
    s_document_term: pd.DataFrame
        One of
        :meth:`texthero.representation.tfidf`
        :meth:`texthero.representation.count`
        :meth:`texthero.representation.term_frequency`

    n_words: int, default to 3
        Number of words to fetch per topic, should
        be <= 30.

    Returns
    -------
    Series with the document IDs as index and
    a list of n_words relevant words per
    document as values.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, band, guitar"])
    >>> s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> hero.top_words_per_document(s_tfidf, n_words=2) # doctest: +SKIP
    0       [soccer, sports]
    1    [violin, orchestra]
    2          [fun, sports]
    3         [guitar, band]
    dtype: object
    >>> # We can see that the function tries to
    >>> # find terms that distinguish the documents,
    >>> # so "music" is not chosen for documents
    >>> # 1 and 3 as it's found in both.

    See Also
    --------
    :meth:`top_words_per_topic`

    TODO add tutorial link

    """
    # Create a categorical Series that has
    # one new cluster for every document.
    s_cluster = pd.Series(
        np.arange(len(s_document_term)), index=s_document_term.index, dtype="category"
    )

    # Call top_words_per_topic with the new cluster series
    # (so every document is one distinct "topic")
    s_top_words_per_document = top_words_per_topic(
        s_document_term, s_cluster, n_words=n_words
    )

    return s_top_words_per_document.reindex(s_document_term.index)


"""
import texthero as hero
import pandas as pd
df = pd.read_csv(
    "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)


s_tfidf = df["text"].pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf, max_df = 0.5, min_df = 100)
s_lda = s_tfidf.pipe(hero.truncatedSVD, n_components=5)
# a, b = hero.visualize_topics(s_tfidf, s_lda)
hero.browser_display(hero.visualize_topics(s_tfidf, s_lda))

"""
