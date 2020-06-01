---
id: texthero.preprocessing.get_default_pipeline
title: preprocessing.get_default_pipeline
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-get-default-pipeline">
<h1>texthero.preprocessing.get_default_pipeline<a class="headerlink" href="#texthero-preprocessing-get-default-pipeline" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.get_default_pipeline">
<code class="sig-name descname">get_default_pipeline</code><span class="sig-paren">(</span><span class="sig-paren">)</span> → List<span class="p">[</span>Callable<span class="p">[</span><span class="p">[</span>pandas.core.series.Series<span class="p">]</span><span class="p">, </span>pandas.core.series.Series<span class="p">]</span><span class="p">]</span><a class="headerlink" href="#texthero.preprocessing.get_default_pipeline" title="Permalink to this definition">¶</a></dt>
<dd><p>Return a list contaning all the methods used in the default cleaning pipeline.</p>
<dl class="simple">
<dt>Return a list with the following functions:</dt><dd><ol class="arabic simple">
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