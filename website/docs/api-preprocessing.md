---
id: api-preprocessing
title: Preprocessing
hide_title: false
---

<div>
<span class="target" id="module-texthero.preprocessing"></span><p>Preprocess text-based Pandas DataFrame</p>
<div class="section" id="examples">
<h1>Examples<a class="headerlink" href="#examples" title="Permalink to this headline">¶</a></h1>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">pipeline</span><span class="p">()</span> <span class="o">...</span>
</pre></div>
</div>
<table class="longtable table">
<colgroup>
<col style="width: 10%"/>
<col style="width: 90%"/>
</colgroup>
<tbody>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.clean.html#texthero.preprocessing.clean" title="texthero.preprocessing.clean"><code class="xref py py-obj docutils literal notranslate"><span class="pre">clean</span></code></a>(s[, pipeline])</p></td>
<td><p>Clean pandas series by appling a preprocessing pipeline.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.drop_no_content.html#texthero.preprocessing.drop_no_content" title="texthero.preprocessing.drop_no_content"><code class="xref py py-obj docutils literal notranslate"><span class="pre">drop_no_content</span></code></a>(s)</p></td>
<td><p>Drop all rows where has_content is empty.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.get_default_pipeline.html#texthero.preprocessing.get_default_pipeline" title="texthero.preprocessing.get_default_pipeline"><code class="xref py py-obj docutils literal notranslate"><span class="pre">get_default_pipeline</span></code></a>()</p></td>
<td><p>Return a list contaning all the methods used in the default cleaning pipeline.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.has_content.html#texthero.preprocessing.has_content" title="texthero.preprocessing.has_content"><code class="xref py py-obj docutils literal notranslate"><span class="pre">has_content</span></code></a>(s)</p></td>
<td><p>For each row, check that there is content.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_angle_brackets.html#texthero.preprocessing.remove_angle_brackets" title="texthero.preprocessing.remove_angle_brackets"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_angle_brackets</span></code></a>(s)</p></td>
<td><p>Remove content within angle brackets &lt;&gt; and the angle brackets.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_brackets.html#texthero.preprocessing.remove_brackets" title="texthero.preprocessing.remove_brackets"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_brackets</span></code></a>(s)</p></td>
<td><p>Remove content within brackets and the brackets.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_curly_brackets.html#texthero.preprocessing.remove_curly_brackets" title="texthero.preprocessing.remove_curly_brackets"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_curly_brackets</span></code></a>(s)</p></td>
<td><p>Remove content within curly brackets {} and the curly brackets.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_diacritics.html#texthero.preprocessing.remove_diacritics" title="texthero.preprocessing.remove_diacritics"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_diacritics</span></code></a>(input)</p></td>
<td><p>Remove all diacritics.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_digits.html#texthero.preprocessing.remove_digits" title="texthero.preprocessing.remove_digits"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_digits</span></code></a>(input[, only_blocks])</p></td>
<td><p>Remove all digits from a series and replace it with an empty space.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_html_tags.html#texthero.preprocessing.remove_html_tags" title="texthero.preprocessing.remove_html_tags"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_html_tags</span></code></a>(s)</p></td>
<td><p>Remove html tags from the given Pandas Series.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_punctuation.html#texthero.preprocessing.remove_punctuation" title="texthero.preprocessing.remove_punctuation"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_punctuation</span></code></a>(input)</p></td>
<td><p>Remove string.punctuation (!”#$%&amp;’()*+,-./:;&lt;=&gt;?@[]^_`{|}~).</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_round_brackets.html#texthero.preprocessing.remove_round_brackets" title="texthero.preprocessing.remove_round_brackets"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_round_brackets</span></code></a>(s)</p></td>
<td><p>Remove content within parentheses () and parentheses.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_square_brackets.html#texthero.preprocessing.remove_square_brackets" title="texthero.preprocessing.remove_square_brackets"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_square_brackets</span></code></a>(s)</p></td>
<td><p>Remove content within square brackets [] and the square brackets.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_stopwords.html#texthero.preprocessing.remove_stopwords" title="texthero.preprocessing.remove_stopwords"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_stopwords</span></code></a>(input, stopwords, …)</p></td>
<td><p>Remove all instances of <cite>words</cite> and replace it with an empty space.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_urls.html#texthero.preprocessing.remove_urls" title="texthero.preprocessing.remove_urls"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_urls</span></code></a>(s)</p></td>
<td><p>Remove all urls from a given Series.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_whitespace.html#texthero.preprocessing.remove_whitespace" title="texthero.preprocessing.remove_whitespace"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_whitespace</span></code></a>(input)</p></td>
<td><p>Remove all extra white spaces between words.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.replace_punctuation.html#texthero.preprocessing.replace_punctuation" title="texthero.preprocessing.replace_punctuation"><code class="xref py py-obj docutils literal notranslate"><span class="pre">replace_punctuation</span></code></a>(input, symbol)</p></td>
<td><p>Replace string.punctuation (!”#$%&amp;’()*+,-./:;&lt;=&gt;?@[]^_`{|}~) with symbol argument.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.replace_stopwords.html#texthero.preprocessing.replace_stopwords" title="texthero.preprocessing.replace_stopwords"><code class="xref py py-obj docutils literal notranslate"><span class="pre">replace_stopwords</span></code></a>(input, symbol, stopwords, …)</p></td>
<td><p>Replace all stopwords with symbol.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.stem.html#texthero.preprocessing.stem" title="texthero.preprocessing.stem"><code class="xref py py-obj docutils literal notranslate"><span class="pre">stem</span></code></a>(input[, stem, language])</p></td>
<td><p>Stem series using either ‘porter’ or ‘snowball’ NLTK stemmers.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.tokenize.html#texthero.preprocessing.tokenize" title="texthero.preprocessing.tokenize"><code class="xref py py-obj docutils literal notranslate"><span class="pre">tokenize</span></code></a>(s)</p></td>
<td><p>Tokenize each row of the given Series.</p></td>
</tr>
</tbody>
</table>
</div>
</div>