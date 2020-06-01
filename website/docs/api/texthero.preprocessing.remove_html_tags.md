---
id: texthero.preprocessing.remove_html_tags
title: preprocessing.remove_html_tags
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-remove-html-tags">
<h1>texthero.preprocessing.remove_html_tags<a class="headerlink" href="#texthero-preprocessing-remove-html-tags" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.remove_html_tags">
<code class="sig-name descname">remove_html_tags</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.remove_html_tags" title="Permalink to this definition">¶</a></dt>
<dd><p>Remove html tags from the given Pandas Series.</p>
<p>Remove all html tags of the type <cite>&lt;.*?&gt;</cite> such as &lt;html&gt;, &lt;p&gt;, &lt;div class=”hello”&gt; and remove all html tags of type &amp;nbsp and return a cleaned Pandas Series.</p>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"&lt;html&gt;&lt;h1&gt;Title&lt;/h1&gt;&lt;/html&gt;"</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">remove_html_tags</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    Title</span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>