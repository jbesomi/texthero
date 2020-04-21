"""
Text representation

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

def do_tfidf(df, text_columns, max_features=100):
    def do_tfidf_col(docs):
        tfidf = TfidfVectorizer(use_idf=True, max_features=max_features)
        return tfidf.fit_transform(docs).toarray().tolist()

    if isinstance(text_columns, str):
        df['tfidf_' + text_columns] = do_tfidf_col(list(df[text_columns]))

    else:
        for col in text_columns:
            df['tfidf_' + col] = do_tfidf_col(list(df[col]))

    return df


def do_pca(df, vector_columns, n_components=2):
    def do_pca_col(vectors):
        pca = PCA(n_components=n_components)
        return pca.fit_transform(vectors).tolist()

    if isinstance(vector_columns, str):
        df['pca_' + vector_columns] = do_pca_col(list(df[vector_columns]))

    else:
        for col in vector_columns:
            df['pca_' + col] = do_pca_col(list(df[col]))

    return df


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
