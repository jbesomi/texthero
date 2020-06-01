"""
Visualize insights and statistics of a text-based Pandas DataFrame.
"""

import pandas as pd
import plotly.express as px

from wordcloud import WordCloud
from nltk import NLTKWordTokenizer

from texthero import preprocessing
import string

# from typing import Boolean


@pd.api.extensions.register_series_accessor("words")
class WordsAccessor:
    """
    To access plot directly from a series.

    This is just for testing.

    Example
    -------
    df['text'].words.plot()

    
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        # keeping as an example
        return
        if "pca" not in obj.columns:
            raise AttributeError("Must have 'latitude' and 'longitude'.")

    @property
    def center(self):
        # return the geographic center point of this DataFrame
        # keeping as an example
        return
        lat = self._obj.latitude
        lon = self._obj.longitude
        return (float(lon.mean()), float(lat.mean()))

    def plot(self, num_words=20, a=None):

        top = top_words(self._obj)
        df = px.data.tips()
        fig = px.bar(x=top[:num_words].index, y=top[:num_words].values)
        fig.update_traces(
            marker=dict(colorscale="Portland", color=top[:num_words].values)
        )
        fig.show()


def scatterplot(
    df: pd.DataFrame,
    col: str,
    color: str = None,
    hover_data: [] = None,
    title="",
    return_figure=False,
):
    """
    Show scatterplot using python plotly scatter.

    Parameters
    ----------
    df
    col
        The name of the column of the DataFrame used for x and y axis.
    """

    pca0 = df[col].apply(lambda x: x[0])
    pca1 = df[col].apply(lambda x: x[1])

    fig = px.scatter(
        df, x=pca0, y=pca1, color=color, hover_data=hover_data, title=title
    )
    # fig.show(config={'displayModeBar': False})
    fig.show()

    if return_figure:
        return fig


"""

"""


def wordcloud(
    s: pd.Series,
    font_path: str = None,
    width: int = 400,
    height: int = 200,
    margin=2,
    ranks_only=None,
    prefer_horizontal=0.9,
    mask=None,
    scale=1,
    color_func=None,
    max_words=200,
    min_font_size=4,
    stopwords=None,
    random_state=None,
    background_color="black",
    max_font_size=None,
    font_step=1,
    mode="RGB",
    relative_scaling="auto",
    regexp=None,
    collocations=True,
    colormap=None,
    normalize_plurals=True,
    contour_width=0,
    contour_color="black",
    repeat=False,
    include_numbers=False,
    min_word_length=0,
    collocation_threshold=30,
    return_figure=False,
):
    """
    Plot wordcloud image using WordCloud from word_cloud package.

    Most of the arguments are very similar if not equal to the mother function. In constrast, all words are taken into account when computing the wordcloud, inclusive stopwords. They can be easily removed with preprocessing.remove_stopwords.

    Word are compute using generate_from_frequencies.

    Parameters
    ----------
    s : pd.Series
    font_path : str
        Font path to the font that will be used (OTF or TTF). Defaults to DroidSansMono path on a Linux machine. If you are on another OS or don't have this font, you need to adjust this path.
    width : int
        Width of the canvas.
    height : int
        Height of the canvas.
    max_words : number (default=200)
        The maximum number of words.
    mask : nd-array or None (default=None)
        When set, gives a binary mask on where to draw words. When set, width and height will be ignored and the shape of mask will be used instead. All white (#FF or #FFFFFF) entries will be considerd "masked out" while other entries will be free to draw on.
    contour_width: float (default=0)
        If mask is not None and contour_width > 0, draw the mask contour.
    contour_color: color value (default="black")
        Mask contour color.
    min_font_size : int (default=4)
        Smallest font size to use. Will stop when there is no more room in this size.
    background_color : color value (default="black")
        Background color for the word cloud image.
    max_font_size : int or None (default=None)
        Maximum font size for the largest word. If None, height of the image is used.
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
    # text = s.str.cat(sep=" ")

    wordcloud = WordCloud(
        background_color="white",
        min_font_size=10,
        stopwords=[],  # will use generate from frequencies.
        normalize_plurals=False,
    ).generate_from(text)

    fig = px.imshow(wordcloud, title=title)
    fig.show()

    if return_figure:
        return fig


def top_words(s: pd.Series, normalize=False) -> pd.Series:
    r"""
    Return a pandas series with index the top words and as value the count.

    Tokenization: split by space and remove all punctuations that are not between characters.
    
    Parameters
    ----------
    normalize :
        When set to true, return normalized values.

    """

    # Replace all punctuation that are NOT in-between chacarters
    # This means, they have either a non word-bounding \B, are at the start ^, or at the end $
    # As re.sub replace all and not just the matching group, add matching parenthesis to the character
    # to keep during replacement.
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
