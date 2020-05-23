"""
Visualize insights and statistics of a text-based Pandas DataFrame.
"""

import pandas as pd
import plotly.express as px

from wordcloud import WordCloud
from nltk import NLTKWordTokenizer

from texthero import preprocessing
import string


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


def wordcloud(s: pd.Series, title="", return_figure=False):
    """
    Show wordcloud using WordCloud.

    Parameters
    ----------
    df
    col
        The name of the column of the DataFrame containing the text data.

    """
    text = s.str.cat(sep=" ")

    wordcloud = WordCloud(background_color="white", min_font_size=10).generate(text)

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
