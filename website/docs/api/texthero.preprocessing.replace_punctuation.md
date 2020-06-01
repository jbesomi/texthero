---
id: texthero.preprocessing.replace_punctuation
title: preprocessing.replace_punctuation
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-replace-punctuation">
<h1>texthero.preprocessing.replace_punctuation<a class="headerlink" href="#texthero-preprocessing-replace-punctuation" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.replace_punctuation">
<code class="sig-name descname">replace_punctuation</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">symbol</span><span class="p">:</span> <span class="n"><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.8)">str</a></span> <span class="o">=</span> <span class="default_value">' '</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.replace_punctuation" title="Permalink to this definition">¶</a></dt>
<dd><p>Replace all punctuation with a given symbol.</p>
<p><cite>replace_punctuation</cite> replace all punctuation from the given Pandas Series and replace it with a custom symbol. It consider as punctuation characters all <a class="reference external" href="https://docs.python.org/3/library/string.html#string.punctuation" title="(in Python v3.8)"><code class="xref py py-data docutils literal notranslate"><span class="pre">string.punctuation</span></code></a> symbols <cite>!”#$%&amp;’()*+,-./:;&lt;=&gt;?@[]^_`{|}~).</cite></p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><dl class="simple">
<dt><strong>input</strong><span class="classifier">Pandas Series</span></dt><dd></dd>
<dt><strong>symbol</strong><span class="classifier">str (default single empty space)</span></dt><dd><p>Symbol to use as replacement for all string punctuation.</p>
</dd>
</dl>
</dd>
</dl>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"Finnaly."</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">replace_punctuation</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="s2">" &lt;PUNCT&gt; "</span><span class="p">)</span>
<span class="go">0    Finnaly &lt;PUNCT&gt; </span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>