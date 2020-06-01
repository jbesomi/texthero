---
id: texthero.preprocessing.remove_punctuation
title: preprocessing.remove_punctuation
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-remove-punctuation">
<h1>texthero.preprocessing.remove_punctuation<a class="headerlink" href="#texthero-preprocessing-remove-punctuation" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.remove_punctuation">
<code class="sig-name descname">remove_punctuation</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.remove_punctuation" title="Permalink to this definition">¶</a></dt>
<dd><p>Replace all punctuation with a single space (” “).</p>
<p><cite>remove_punctuation</cite> removes all punctuation from the given Pandas Series and replace it with a single space. It consider as punctuation characters all <a class="reference external" href="https://docs.python.org/3/library/string.html#string.punctuation" title="(in Python v3.8)"><code class="xref py py-data docutils literal notranslate"><span class="pre">string.punctuation</span></code></a> symbols <cite>!”#$%&amp;’()*+,-./:;&lt;=&gt;?@[]^_`{|}~).</cite></p>
<p>See also <a class="reference internal" href="texthero.preprocessing.replace_punctuation.html#texthero.preprocessing.replace_punctuation" title="texthero.preprocessing.replace_punctuation"><code class="xref py py-meth docutils literal notranslate"><span class="pre">replace_punctuation()</span></code></a> to replace punctuation with a custom symbol.</p>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"Finnaly."</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">remove_punctuation</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    Finnaly </span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>