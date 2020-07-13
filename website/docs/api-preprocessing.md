---
id: api-preprocessing
title: Preprocessing
hide_title: false
---

<div>
<span class="target" id="module-texthero.preprocessing"></span><p>The texthero.preprocess module allow for efficient pre-processing of text-based Pandas Series and DataFrame.</p>
<table class="longtable table">
<colgroup>
<col style="width: 10%"/>
<col style="width: 90%"/>
</colgroup>
<tbody>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.clean.html#texthero.preprocessing.clean" title="texthero.preprocessing.clean"><code class="xref py py-obj docutils literal notranslate"><span class="pre">clean</span></code></a>(s[, pipeline])</p></td>
<td><p>Pre-process a text-based Pandas Series.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.drop_no_content.html#texthero.preprocessing.drop_no_content" title="texthero.preprocessing.drop_no_content"><code class="xref py py-obj docutils literal notranslate"><span class="pre">drop_no_content</span></code></a>(s)</p></td>
<td><p>Drop all rows without content.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.get_default_pipeline.html#texthero.preprocessing.get_default_pipeline" title="texthero.preprocessing.get_default_pipeline"><code class="xref py py-obj docutils literal notranslate"><span class="pre">get_default_pipeline</span></code></a>()</p></td>
<td><p>Return a list contaning all the methods used in the default cleaning pipeline.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.has_content.html#texthero.preprocessing.has_content" title="texthero.preprocessing.has_content"><code class="xref py py-obj docutils literal notranslate"><span class="pre">has_content</span></code></a>(s)</p></td>
<td><p>Return a Boolean Pandas Series indicating if the rows has content.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_angle_brackets.html#texthero.preprocessing.remove_angle_brackets" title="texthero.preprocessing.remove_angle_brackets"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_angle_brackets</span></code></a>(s)</p></td>
<td><p>Remove content within angle brackets &lt;&gt; and the angle brackets.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_brackets.html#texthero.preprocessing.remove_brackets" title="texthero.preprocessing.remove_brackets"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_brackets</span></code></a>(s)</p></td>
<td><p>Remove content within brackets and the brackets itself.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_curly_brackets.html#texthero.preprocessing.remove_curly_brackets" title="texthero.preprocessing.remove_curly_brackets"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_curly_brackets</span></code></a>(s)</p></td>
<td><p>Remove content within curly brackets {} and the curly brackets.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_diacritics.html#texthero.preprocessing.remove_diacritics" title="texthero.preprocessing.remove_diacritics"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_diacritics</span></code></a>(input)</p></td>
<td><p>Remove all diacritics and accents.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_digits.html#texthero.preprocessing.remove_digits" title="texthero.preprocessing.remove_digits"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_digits</span></code></a>(input[, only_blocks])</p></td>
<td><p>Remove all digits and replace it with a single space.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_html_tags.html#texthero.preprocessing.remove_html_tags" title="texthero.preprocessing.remove_html_tags"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_html_tags</span></code></a>(s)</p></td>
<td><p>Remove html tags from the given Pandas Series.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_punctuation.html#texthero.preprocessing.remove_punctuation" title="texthero.preprocessing.remove_punctuation"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_punctuation</span></code></a>(input)</p></td>
<td><p>Replace all punctuation with a single space (” “).</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_round_brackets.html#texthero.preprocessing.remove_round_brackets" title="texthero.preprocessing.remove_round_brackets"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_round_brackets</span></code></a>(s)</p></td>
<td><p>Remove content within parentheses () and parentheses.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_square_brackets.html#texthero.preprocessing.remove_square_brackets" title="texthero.preprocessing.remove_square_brackets"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_square_brackets</span></code></a>(s)</p></td>
<td><p>Remove content within square brackets [] and the square brackets.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_stopwords.html#texthero.preprocessing.remove_stopwords" title="texthero.preprocessing.remove_stopwords"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_stopwords</span></code></a>(input, stopwords, …[, …])</p></td>
<td><p>Remove all instances of <cite>words</cite>.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_urls.html#texthero.preprocessing.remove_urls" title="texthero.preprocessing.remove_urls"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_urls</span></code></a>(s)</p></td>
<td><p>Remove all urls from a given Pandas Series.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.replace_urls.html#texthero.preprocessing.replace_urls" title="texthero.preprocessing.replace_urls"><code class="xref py py-obj docutils literal notranslate"><span class="pre">replace_urls</span></code></a>(s, symbol)</p></td>
<td><p>Replace all urls with the given symbol.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.remove_whitespace.html#texthero.preprocessing.remove_whitespace" title="texthero.preprocessing.remove_whitespace"><code class="xref py py-obj docutils literal notranslate"><span class="pre">remove_whitespace</span></code></a>(input)</p></td>
<td><p>Remove any extra white spaces.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.replace_punctuation.html#texthero.preprocessing.replace_punctuation" title="texthero.preprocessing.replace_punctuation"><code class="xref py py-obj docutils literal notranslate"><span class="pre">replace_punctuation</span></code></a>(input, symbol)</p></td>
<td><p>Replace all punctuation with a given symbol.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="api/texthero.preprocessing.replace_stopwords.html#texthero.preprocessing.replace_stopwords" title="texthero.preprocessing.replace_stopwords"><code class="xref py py-obj docutils literal notranslate"><span class="pre">replace_stopwords</span></code></a>(input, symbol, stopwords, …)</p></td>
<td><p>Replace all instances of <cite>words</cite> with symbol.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="api/texthero.preprocessing.tokenize.html#texthero.preprocessing.tokenize" title="texthero.preprocessing.tokenize"><code class="xref py py-obj docutils literal notranslate"><span class="pre">tokenize</span></code></a>(s)</p></td>
<td><p>Tokenize each row of the given Series.</p></td>
</tr>
</tbody>
</table>
</div>