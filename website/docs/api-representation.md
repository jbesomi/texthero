---
id: api-representation 
title: Representation
---

Map words into vectors using different algorithms such as TF-IDF, word2vec or GloVe.


### texthero.representation.do_count(s, max_features=100)
Represent input on a Count vector space.


### texthero.representation.do_dbscan(s, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
Perform DBSCAN clustering.


### texthero.representation.do_kmeans(s, n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=- 1, algorithm='auto')
Perform K-means clustering algorithm.


### texthero.representation.do_meanshift(s, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300)
Perform mean shift clustering.


### texthero.representation.do_nmf(s, n_components=2)
Perform non-negative matrix factorization.


### texthero.representation.do_pca(s, n_components=2)
Perform PCA.


### texthero.representation.do_tfidf(s, max_features=100)
Represent input on a TF-IDF vector space.


### texthero.representation.do_tsne(s, vector_columns, n_components, perplexity, early_exaggeration, learning_rate, n_iter)
Perform TSNE.
