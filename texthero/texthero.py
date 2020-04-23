from . import preprocessing
from . import representation
from . import visualization

import pandas as pd

"""
Preprocessing
"""

def do_preprocess(df, text_columns='text', pipeline=None):
    return preprocessing.do_preprocess(df,text_columns, pipeline)

"""
Representation
"""

def do_tfidf(df, text_columns='text', max_features=100):
    return representation.do_tfidf(df, text_columns=text_columns, max_features=max_features)

def do_pca(df, vector_columns='tfidf_text', n_components=2):
    return representation.do_pca(df, vector_columns, n_components)

def do_nmf(df, vector_columns='tfidf_text', n_components=2):
    return representation.do_nmf(df, vector_columns, n_components)

def do_tsne(df, vector_columns, n_components=2, perplexity=30, early_exaggeration=12, learning_rate=200, n_iter=1000):
    return representation.do_tsne(df,
                        vector_columns,
                        n_components,
                        perplexity,
                        early_exaggeration,
                        learning_rate,
                        n_iter)

"""
Visualization
"""
def scatterplot(df, column, color=None, hover_data=None):
    return visualization.scatterplot(df, column, color, hover_data)

def top_words(s: pd.Series, normalize=False):
    return visualization.top_words(s, normalize=normalize)
