---
id: api-representation
title: Representation
hide_title: false
---

<div>
<span class="target" id="module-texthero.representation"></span><p>Map words into vectors using different algorithms such as TF-IDF, word2vec or GloVe.</p>
<table class="longtable table">
<colgroup>
<col style="width: 10%"/>
<col style="width: 90%"/>
</colgroup>
<tbody>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.representation.dbscan.html#texthero.representation.dbscan" title="texthero.representation.dbscan"><code class="xref py py-obj docutils literal notranslate"><span class="pre">dbscan</span></code></a>(s[, eps, min_samples, metric, …])</p></td>
<td><p>Perform DBSCAN clustering.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.representation.kmeans.html#texthero.representation.kmeans" title="texthero.representation.kmeans"><code class="xref py py-obj docutils literal notranslate"><span class="pre">kmeans</span></code></a>(s[, n_clusters, init, n_init, …])</p></td>
<td><p>Perform K-means clustering algorithm.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.representation.meanshift.html#texthero.representation.meanshift" title="texthero.representation.meanshift"><code class="xref py py-obj docutils literal notranslate"><span class="pre">meanshift</span></code></a>(s[, bandwidth, seeds, …])</p></td>
<td><p>Perform mean shift clustering.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.representation.nmf.html#texthero.representation.nmf" title="texthero.representation.nmf"><code class="xref py py-obj docutils literal notranslate"><span class="pre">nmf</span></code></a>(s[, n_components])</p></td>
<td><p>Perform non-negative matrix factorization.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.representation.pca.html#texthero.representation.pca" title="texthero.representation.pca"><code class="xref py py-obj docutils literal notranslate"><span class="pre">pca</span></code></a>(s[, n_components])</p></td>
<td><p>Perform principal component analysis on the given Pandas Series.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.representation.term_frequency.html#texthero.representation.term_frequency" title="texthero.representation.term_frequency"><code class="xref py py-obj docutils literal notranslate"><span class="pre">term_frequency</span></code></a>(s, max_features, NoneType] = None)</p></td>
<td><p>Represent a text-based Pandas Series using term_frequency.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.representation.tfidf.html#texthero.representation.tfidf" title="texthero.representation.tfidf"><code class="xref py py-obj docutils literal notranslate"><span class="pre">tfidf</span></code></a>(s[, max_features, min_df, …])</p></td>
<td><p>Represent a text-based Pandas Series using TF-IDF.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.representation.tsne.html#texthero.representation.tsne" title="texthero.representation.tsne"><code class="xref py py-obj docutils literal notranslate"><span class="pre">tsne</span></code></a>(s[, n_components, perplexity, …])</p></td>
<td><p>Perform TSNE on the given pandas series.</p></td>
</tr>
</tbody>
</table>
</div>