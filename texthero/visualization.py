"""
Visualize insights and statistics of a text-based Pandas DataFrame.
"""

import pandas as pd
import numpy as np
import plotly.express as px

from wordcloud import WordCloud

from texthero import preprocessing, representation
from texthero._types import TextSeries, InputSeries
import string

from matplotlib.colors import LinearSegmentedColormap as lsg
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import normalize as sklearn_normalize

import pyLDAvis

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

    color: str, optional, default=None
        Name of the column to use for coloring (rows with same value get same
        color).

    hover_name: str, optional, default=None
        Name of the column to supply title of hover data when hovering over a
        point.

    hover_data: List[str], optional, default=[]
        List of column names to supply data when hovering over a point.

    title: str, default to "".
        Title of the plot.

    return_figure: bool, optional, default=False
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
    ... )
    >>> df["topics"] = (
    ...                hero.tfidf(df["texts"])
    ...                    .pipe(hero.kmeans, n_clusters=2)
    ... )
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

    font_path : str, optional, default=None
        Font path to the font that will be used (OTF or TTF). Defaults to
        DroidSansMono path on a Linux machine. If you are on another OS or
        don't have this font, you need to adjust this path.

    width : int, optional, default=400
        Width of the canvas.

    height : int, optional, default=200
        Height of the canvas.

    max_words : int, optional, default=200
        The maximum number of words.

    mask : nd-array or None, optional, default=None
        When set, gives a binary mask on where to draw words. When set, width
        and height will be ignored and the shape of mask will be used instead.
        All white (#FF or #FFFFFF) entries will be considerd "masked out"
        while other entries will be free to draw on.

    contour_width: float, optional, default=0
        If mask is not None and contour_width > 0, draw the mask contour.

    contour_color: str, optional, default="PAPAYAWHIP"
        Mask contour color.

    min_font_size : int, optional, default=4
        Smallest font size to use. Will stop when there is no more room in
        this size.

    background_color : str, optional, default="PAPAYAWHIP"
        Background color for the word cloud image.

    max_font_size : int or None, optional, default=None
        Maximum font size for the largest word. If None, height of the image
        is used.

    relative_scaling : float, optional, default="auto"
        Importance of relative word frequencies for font-size.  With
        relative_scaling=0, only word-ranks are considered.  With
        relative_scaling=1, a word that is twice as frequent will have twice
        the size.  If you want to consider the word frequencies and not only
        their rank, relative_scaling around .5 often looks good.
        If 'auto' it will be set to 0.5 unless repeat is true, in which
        case it will be set to 0.

    colormap : string or matplotlib colormap, optional, default="viridis"
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
    normalize : bool, optional, default=False.
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


def visualize_topics(
    s_document_term: pd.DataFrame,
    s_document_topic: pd.Series,
    notebook=True,
    return_figure=False,
):
    """
    Combine several Texthero functions to get a
    `pyLDAvis <https://github.com/bmabey/pyLDAvis>`_  visualization
    straight from document_term_matrix and document_topic_matrix.

    Using this function is equivalent to doing the following:
    ```python

    >>> import pyLDAvis  # doctest: +SKIP
    >>> s_document_topic, s_topic_term = hero.topic_matrices(s_document_term, s_document_topic) # doctest: +SKIP
    >>> s_document_topic_distribution = hero.normalize(s_document_topic, norm="l1") # doctest: +SKIP
    >>> s_topic_term_distribution = hero.normalize(s_topic_term, norm="l1") # doctest: +SKIP
    >>> figure = hero.relevant_words_per_topic(s_document_term, s_document_topic_distribution, s_topic_term_distribution, return_figure=True) # doctest: +SKIP
    >>> # in a Jupyter Notebook
    >>> pyLDAvis.display(figure) # doctest: +SKIP
    >>> # otherwise
    >>> pyLDAvis.show(figure) # doctest: +SKIP
    ```

    First input has
    to be output of one of 
    - :meth:`texthero.representation.tfidf`
    - :meth:`texthero.representation.count`
    - :meth:`texthero.representation.term_frequency`.

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
    of topics to terms, using :meth:`hero.topic_matrices`_
    and :meth:`hero.representation.normalize`_.

    These distributions are passed to
    :meth:`hero.relevant_words_per_topic`_, which
    uses `pyLDAvis <https://pyldavis.readthedocs.io/en/latest/>`_
    to visualize the topics and terms.

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

    notebook : bool, default True
        Whether to show the visualization inside
        a Jupyter Notebook or open a new browser tab.
        Set this to False when not inside a Jupyter Notebook.
    return_figure : bool, default False
        Whether to only return the figure instead
        of showing it.

    Examples
    --------
    Using Clustering:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, band, guitar"])
    >>> s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> s_cluster = s_tfidf.pipe(hero.normalize).pipe(hero.pca, n_components=2).pipe(hero.kmeans, n_clusters=2)
    >>> # Display in a new browser window:
    >>> hero.visualize_topics(s_tfidf, s_cluster, notebook=False) # doctest: +SKIP
    >>> # Display inside the current Jupyter Notebook:
    >>> hero.visualize_topics(s_tfidf, s_cluster, notebook=True) # doctest: +SKIP

    Using LDA:

    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Football, Sports, Soccer", "music, violin, orchestra", "football, fun, sports", "music, band, guitar"])
    >>> s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf)
    >>> s_lda = s_tfidf.pipe(hero.lda, n_components=2)
    >>> # Display in a new browser window:
    >>> hero.visualize_topics(s_tfidf, s_cluster, notebook=False) # doctest: +SKIP
    >>> # Display inside the current Jupyter Notebook:
    >>> hero.visualize_topics(s_tfidf, s_cluster, notebook=True) # doctest: +SKIP

    See Also
    --------
    `pyLDAvis <https://pyldavis.readthedocs.io/en/latest/>`_
    for the methodology on how to find relevant terms.

    :meth:`texthero.representation.topic_matrices`_

    :meth:`texthero.representation.relevant_words_per_topic`_

    TODO add tutorial link

    """
    # Get topic matrices.
    s_document_topic, s_topic_term = representation.topic_matrices(
        s_document_term, s_document_topic
    )

    # Get topic distributions through normalization.
    s_document_topic_distribution = representation.normalize(
        s_document_topic, norm="l1"
    )
    s_topic_term_distribution = representation.normalize(s_topic_term, norm="l1")

    # Get the pyLDAvis figure.
    figure = representation.relevant_words_per_topic(
        s_document_term,
        s_document_topic_distribution,
        s_topic_term_distribution,
        return_figure=True,
    )

    if return_figure:
        return figure

    # Visualize it.
    if notebook:
        # Import here as non-notebook users don't have this.
        import IPython
        return IPython.display.display(pyLDAvis.display(figure))
    else:
        pyLDAvis.show(figure)
