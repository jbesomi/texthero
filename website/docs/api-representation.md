---
id: api-representation 
title: Representation
---

Map words into vectors using different algorithms such as TF-IDF, word2vec or GloVe.


### texthero.representation.dbscan(s, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
Perform DBSCAN clustering.


### texthero.representation.kmeans(s, n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=- 1, algorithm='auto')
Perform K-means clustering algorithm.


### texthero.representation.meanshift(s, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300)
Perform mean shift clustering.


### texthero.representation.nmf(s, n_components=2)
Perform non-negative matrix factorization.


### texthero.representation.pca(s, n_components=2)
Perform PCA.


### texthero.representation.term_frequency(s, max_features=None, lowercase=False, token_pattern='\\\\S+')
Represent input on term frequency.


### texthero.representation.tfidf(s, max_features=None, min_df=1, token_pattern='\\\\S+', lowercase=False)
Represent input on a TF-IDF vector space.


### texthero.representation.tsne(s, n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=- 1)
Perform TSNE.
