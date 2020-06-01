---
id: texthero.preprocessing.drop_no_content
title: preprocessing.drop_no_content
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-drop-no-content">
<h1>texthero.preprocessing.drop_no_content<a class="headerlink" href="#texthero-preprocessing-drop-no-content" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.drop_no_content">
<code class="sig-name descname">drop_no_content</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em><span class="sig-paren">)</span><a class="headerlink" href="#texthero.preprocessing.drop_no_content" title="Permalink to this definition">¶</a></dt>
<dd><p>Drop all rows without content.</p>
<p>Drop all rows from the given Pandas Series where <a class="reference internal" href="texthero.preprocessing.has_content.html#texthero.preprocessing.has_content" title="texthero.preprocessing.has_content"><code class="xref py py-meth docutils literal notranslate"><span class="pre">has_content()</span></code></a> is False.</p>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">([</span><span class="s2">"content"</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">nan</span><span class="p">,</span> <span class="s2">"</span><span class="se">\t\n</span><span class="s2">"</span><span class="p">,</span> <span class="s2">" "</span><span class="p">])</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">drop_no_content</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    content</span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>