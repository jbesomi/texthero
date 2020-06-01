---
id: texthero.preprocessing.clean
title: preprocessing.clean
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-clean">
<h1>texthero.preprocessing.clean<a class="headerlink" href="#texthero-preprocessing-clean" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.clean">
<code class="sig-name descname">clean</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">pipeline</span><span class="o">=</span><span class="default_value">None</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.clean" title="Permalink to this definition">¶</a></dt>
<dd><p>Pre-process a text-based Pandas Series.</p>
<dl class="simple">
<dt>Default pipeline:</dt><dd><ol class="arabic simple">
<li><p><code class="xref py py-meth docutils literal notranslate"><span class="pre">texthero.preprocessing.fillna()</span></code></p></li>
<li><p><code class="xref py py-meth docutils literal notranslate"><span class="pre">texthero.preprocessing.lowercase()</span></code></p></li>
<li><p><a class="reference internal" href="texthero.preprocessing.remove_digits.html#texthero.preprocessing.remove_digits" title="texthero.preprocessing.remove_digits"><code class="xref py py-meth docutils literal notranslate"><span class="pre">texthero.preprocessing.remove_digits()</span></code></a></p></li>
<li><p><a class="reference internal" href="texthero.preprocessing.remove_punctuation.html#texthero.preprocessing.remove_punctuation" title="texthero.preprocessing.remove_punctuation"><code class="xref py py-meth docutils literal notranslate"><span class="pre">texthero.preprocessing.remove_punctuation()</span></code></a></p></li>
<li><p><a class="reference internal" href="texthero.preprocessing.remove_diacritics.html#texthero.preprocessing.remove_diacritics" title="texthero.preprocessing.remove_diacritics"><code class="xref py py-meth docutils literal notranslate"><span class="pre">texthero.preprocessing.remove_diacritics()</span></code></a></p></li>
<li><p><a class="reference internal" href="texthero.preprocessing.remove_stopwords.html#texthero.preprocessing.remove_stopwords" title="texthero.preprocessing.remove_stopwords"><code class="xref py py-meth docutils literal notranslate"><span class="pre">texthero.preprocessing.remove_stopwords()</span></code></a></p></li>
<li><p><a class="reference internal" href="texthero.preprocessing.remove_whitespace.html#texthero.preprocessing.remove_whitespace" title="texthero.preprocessing.remove_whitespace"><code class="xref py py-meth docutils literal notranslate"><span class="pre">texthero.preprocessing.remove_whitespace()</span></code></a></p></li>
</ol>
</dd>
</dl>
</dd></dl>
</div>
</div>