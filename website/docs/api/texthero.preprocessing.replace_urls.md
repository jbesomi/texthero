---
id: texthero.preprocessing.replace_urls
title: preprocessing.replace_urls
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-replace-urls">
<h1>texthero.preprocessing.replace_urls<a class="headerlink" href="#texthero-preprocessing-replace-urls" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.replace_urls">
<code class="sig-name descname">replace_urls</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">symbol</span><span class="p">:</span> <span class="n"><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.8)">str</a></span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.replace_urls" title="Permalink to this definition">¶</a></dt>
<dd><p>Replace all urls with the given symbol.</p>
<p><cite>replace_urls</cite> replace any urls from the given Pandas Series with the given symbol.</p>
<div class="alert alert-info">
<p class="admonition-title">See also</p>
<dl class="simple">
<dt><a class="reference internal" href="texthero.preprocessing.remove_urls.html#texthero.preprocessing.remove_urls" title="texthero.preprocessing.remove_urls"><code class="xref py py-meth docutils literal notranslate"><span class="pre">texthero.preprocessing.remove_urls()</span></code></a></dt><dd></dd>
</dl>
</div>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"Go to: https://example.com"</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">replace_urls</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="s2">"&lt;URL&gt;"</span><span class="p">)</span>
<span class="go">0    Go to: &lt;URL&gt;</span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>