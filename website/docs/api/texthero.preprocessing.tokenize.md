---
id: texthero.preprocessing.tokenize
title: preprocessing.tokenize
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-tokenize">
<h1>texthero.preprocessing.tokenize<a class="headerlink" href="#texthero-preprocessing-tokenize" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.tokenize">
<code class="sig-name descname">tokenize</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.tokenize" title="Permalink to this definition">¶</a></dt>
<dd><p>Tokenize each row of the given Series.</p>
<p>Tokenize each row of the given Pandas Series and return a Pandas Series where each row contains a list of tokens.</p>
<p>Algorithm: add a space between any punctuation symbol at
exception if the symbol is between two alphanumeric character and split.</p>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">([</span><span class="s2">"Today you're looking great!"</span><span class="p">])</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">tokenize</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    [Today, you're, looking, great, !]</span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>