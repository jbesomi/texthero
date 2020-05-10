"""
Visualize insights and statistics of a text-based Pandas DataFrame.
"""

import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

def scatterplot(df: pd.DataFrame,
                col: str,
                color: str = None,
                hover_data: [] = None,
                title="",
                return_figure=False):
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

    fig = px.scatter(df,
                     x=pca0,
                     y=pca1,
                     color=color,
                     hover_data=hover_data,
                     title=title)
    #fig.show(config={'displayModeBar': False})
    fig.show()

    if return_figure:
        return fig

def wordcloud(s: pd.Series, 
              title="",
              return_figure=False):
    """
    Show wordcloud using WordCloud.

    Parameters
    ----------
    df
    col
        The name of the column of the DataFrame containing the text data.

    """
    text = s.str.cat(sep=' ')
    
    wordcloud = WordCloud(background_color ='white', min_font_size = 10).generate(text) 
    
    fig = px.imshow(wordcloud, title=title)
    fig.show()

    if return_figure:
        return fig


def top_words(s: pd.Series, normalize=False) -> pd.Series:
    """
    Return most common words.

    Parameters
    ----------

    s
    normalize :
        Default is False. If set to True, returns normalized values.

    """
    WHITESPACE_SPLITTER = r"\W+"
    return s.str.split(WHITESPACE_SPLITTER).explode().value_counts(
        normalize=normalize)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
