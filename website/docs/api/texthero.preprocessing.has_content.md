---
id: texthero.preprocessing.has_content
title: preprocessing.has_content
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-has-content">
<h1>texthero.preprocessing.has_content<a class="headerlink" href="#texthero-preprocessing-has-content" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.has_content">
<code class="sig-name descname">has_content</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em><span class="sig-paren">)</span><a class="headerlink" href="#texthero.preprocessing.has_content" title="Permalink to this definition">¶</a></dt>
<dd><p>Return a Boolean Pandas Series indicating if the rows has content.</p>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">([</span><span class="s2">"content"</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">nan</span><span class="p">,</span> <span class="s2">"</span><span class="se">\t\n</span><span class="s2">"</span><span class="p">,</span> <span class="s2">" "</span><span class="p">])</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">has_content</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0     True</span>
<span class="go">1    False</span>
<span class="go">2    False</span>
<span class="go">3    False</span>
<span class="go">dtype: bool</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>