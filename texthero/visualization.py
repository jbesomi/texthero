"""
Text visualization


"""
import pandas as pd
import plotly.express as px

def scatterplot(df: pd.DataFrame , col: str, color: str = None, hover_data: [] = None, title="") -> None:
    """
    Scatterplot of df[column].

    The df[column] must be a tuple of 2d-coordinates.

    Usage example:

        >>> import texthero
        >>> df = pd.DataFrame([(0,1), (1,0)], columns='pca')
        >>> texthero.visualization.scatterplot(df, 'pca')

    """

    pca0 = df[col].apply(lambda x: x[0])
    pca1 = df[col].apply(lambda x: x[1])

    fig = px.scatter(df,
                     x=pca0,
                     y=pca1,
                     color=color,
                     hover_data=hover_data,
                     title=title
                    )

    fig.show(config={'displayModeBar': False})
    return fig

def top_words(s: pd.Series,  normalize=True) -> pd.Series:
    """
    Return most common words of a given series sorted from most used.
    """
    WHITESPACE_SPLITTER = r"\W+"
    return s.str.split(WHITESPACE_SPLITTER).explode().value_counts(normalize=normalize)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
