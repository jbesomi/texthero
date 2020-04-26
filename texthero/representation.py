"""
Text representation

"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

def do_tfidf(s: pd.Series, max_features=100):
    tfidf = TfidfVectorizer(use_idf=True, max_features=max_features)
    return pd.Series(tfidf.fit_transform(s).toarray().tolist())


def do_pca(s, n_components=2):
    pca = PCA(n_components=n_components)
    return pd.Series(pca.fit_transform(list(s)).tolist())


def do_nmf(df, vector_columns, n_components=2):
    def do_nmf_col(vectors):
        nmf = NMF(n_components=n_components, init='random', random_state=0)
        return nmf.fit_transform(vectors).tolist()

    if isinstance(vector_columns, str):
        df['nmf' + vector_columns] = do_nmf_col(list(df[vector_columns]))

    else:
        for col in vector_columns:
            df['nmf_' + col] = do_nmf_col(list(df[col]))

    return df


def do_tsne(df, vector_columns, n_components, perplexity, early_exaggeration, learning_rate, n_iter):
    def do_tsne_col(vectors):
        tsne = TSNE(n_components=n_components,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    learning_rate=learning_rate,
                    n_iter=n_iter
        )
        return tsne.fit_transform(vectors).tolist()

    if isinstance(vector_columns, str):
        df['tsne_' + vector_columns] = do_tsne_col(list(df[vector_columns]))

    else:
        for col in vector_columns:
            df['tsne_' + col] = do_tsne_col(list(df[col]))

    return df

if __name__ == "__main__":
    import doctest
    doctest.testmod()
