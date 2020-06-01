---
id: texthero.preprocessing.remove_whitespace
title: preprocessing.remove_whitespace
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-remove-whitespace">
<h1>texthero.preprocessing.remove_whitespace<a class="headerlink" href="#texthero-preprocessing-remove-whitespace" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.remove_whitespace">
<code class="sig-name descname">remove_whitespace</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.remove_whitespace" title="Permalink to this definition">¶</a></dt>
<dd><p>Remove any extra white spaces.</p>
<p>Remove any extra whitespace in the given Pandas Series. Removes also newline, tabs and any form of space.</p>
<p>Useful when there is a need to visualize a Pandas Series and most cells have many newlines or other kind of space characters.</p>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"Title </span><span class="se">\n</span><span class="s2"> Subtitle </span><span class="se">\t</span><span class="s2">    ..."</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">remove_whitespace</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    Title Subtitle ...</span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>