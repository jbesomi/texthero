---
id: texthero.preprocessing.remove_square_brackets
title: preprocessing.remove_square_brackets
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-remove-square-brackets">
<h1>texthero.preprocessing.remove_square_brackets<a class="headerlink" href="#texthero-preprocessing-remove-square-brackets" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.remove_square_brackets">
<code class="sig-name descname">remove_square_brackets</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em><span class="sig-paren">)</span><a class="headerlink" href="#texthero.preprocessing.remove_square_brackets" title="Permalink to this definition">¶</a></dt>
<dd><p>Remove content within square brackets [] and the square brackets.</p>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"Texthero [is not a superhero!]"</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">remove_square_brackets</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    Texthero </span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>