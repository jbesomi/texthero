---
id: texthero.preprocessing.replace_stopwords
title: preprocessing.replace_stopwords
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-replace-stopwords">
<h1>texthero.preprocessing.replace_stopwords<a class="headerlink" href="#texthero-preprocessing-replace-stopwords" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.replace_stopwords">
<code class="sig-name descname">replace_stopwords</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">symbol</span><span class="p">:</span> <span class="n"><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.8)">str</a></span></em>, <em class="sig-param"><span class="n">stopwords</span><span class="p">:</span> <span class="n">Union<span class="p">[</span>Set<span class="p">[</span><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.8)">str</a><span class="p">]</span><span class="p">, </span>NoneType<span class="p">]</span></span> <span class="o">=</span> <span class="default_value">None</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.replace_stopwords" title="Permalink to this definition">¶</a></dt>
<dd><p>Replace all stopwords with symbol.</p>
<p>By default uses NLTK’s english stopwords of 179 words.</p>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"the book of the jungle"</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">replace_stopwords</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="s2">"X"</span><span class="p">)</span>
<span class="go">0     book   jungle</span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>