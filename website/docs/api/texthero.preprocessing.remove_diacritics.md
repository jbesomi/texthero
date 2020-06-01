---
id: texthero.preprocessing.remove_diacritics
title: preprocessing.remove_diacritics
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-remove-diacritics">
<h1>texthero.preprocessing.remove_diacritics<a class="headerlink" href="#texthero-preprocessing-remove-diacritics" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.remove_diacritics">
<code class="sig-name descname">remove_diacritics</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.remove_diacritics" title="Permalink to this definition">¶</a></dt>
<dd><p>Remove all diacritics and accents.</p>
<p>Remove all diacritics and accents from any word and characters from the given Pandas Series. Return a cleaned version of the Pandas Series.</p>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"Noël means Christmas in French"</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">remove_diacritics</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    Noel means Christmas in French</span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>