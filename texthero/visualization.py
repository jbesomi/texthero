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

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sklearn_normalize
import pyLDAvis

from collections import Counter


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


def plot_topics(s_document_term, s_document_topic, return_figure=False):

    metadata_list = s_document_topic._metadata

    for item in metadata_list:
        if isinstance(item, tuple):
            if item[0] == "vectorizer":
                vectorizer = item[1]
                break
    else:
        # no vectorizer found
        vectorizer = None

    # Get / build matrices from input

    if vectorizer:
        # TODO check sparseness
        # s_document_topic is output of hero.lda or hero.lsi
        document_term_matrix = s_document_term.values
        document_topic_matrix = s_document_topic.values

        topic_term_matrix = vectorizer.components_

    else:
        # s_document_topic is output of some hero clustering function
        indexes_of_unassigned_documents = s_document_topic == -1
        s_document_term = s_document_term[~indexes_of_unassigned_documents]
        s_document_topic = s_document_topic[~indexes_of_unassigned_documents]
        s_document_topic.cat.remove_unused_categories(inplace=True)

        document_term_matrix = s_document_term.sparse.to_coo()

        # Construct document_topic_matrix
        n_rows = len(s_document_topic.index)
        n_cols = len(s_document_topic.values.categories)

        data = [1 for _ in range(n_rows)]
        rows = range(n_rows)
        columns = s_document_topic.values

        document_topic_matrix = csr_matrix(
            (data, (rows, columns)), shape=(n_rows, n_cols)
        )

        topic_term_matrix = document_topic_matrix.T * document_term_matrix

    vocab = list(s_document_term.columns.levels[1])
    doc_lengths = list(s_document_term.sum(axis=1))
    term_frequency = list(s_document_term.sum(axis=0))

    document_topic_distributions = sklearn_normalize(
        document_topic_matrix, norm="l1", axis=1
    )

    topic_term_distributions = sklearn_normalize(topic_term_matrix, norm="l1", axis=1)

    figure = pyLDAvis.prepare(
        **{
            "vocab": vocab,
            "doc_lengths": doc_lengths,
            "term_frequency": term_frequency,
            "doc_topic_dists": document_topic_distributions.toarray().tolist(),
            "topic_term_dists": topic_term_distributions.toarray().tolist(),
        }
    )

    if return_figure:
        return figure
    else:
        pyLDAvis.display(figure)


"""
# plot_topics(s_document_term, s_document_topic)
import pyLDAvis
import pyLDAvis.sklearn
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
docs_raw = newsgroups.data


# TFIDF matrix & vectorizer

tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                stop_words='english',
                                lowercase=True,
                                token_pattern=r'\b[a-zA-Z]{3,}\b',
                                max_df=0.5,
                                min_df=100)

tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())
dtm_tfidf = tfidf_vectorizer.fit_transform(docs_raw)

# LDA

lda_tfidf = LatentDirichletAllocation(n_components=20, random_state=0)
lda_tfidf.fit(dtm_tfidf)


# Prep & Visualize
vis = pyLDAvis.sklearn.prepare(lda_tfidf, dtm_tfidf, tfidf_vectorizer)
pyLDAvis.show(vis)
"""


"""
import texthero as hero
from sklearn.preprocessing import normalize as sklearn_normalize
import pyLDAvis
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

s = pd.Series(newsgroups.data)
s_tfidf = s.pipe(hero.clean).pipe(hero.tokenize).pipe(hero.tfidf, max_df=0.5, min_df=100)
s_cluster = s_tfidf.pipe(hero.pca, n_components=20).pipe(hero.dbscan)

vis = hero.plot_topics(s_tfidf, s_cluster)
pyLDAvis.show(vis)
"""
