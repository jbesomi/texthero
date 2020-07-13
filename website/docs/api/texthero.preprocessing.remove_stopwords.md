---
id: texthero.preprocessing.remove_stopwords
title: preprocessing.remove_stopwords
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-remove-stopwords">
<h1>texthero.preprocessing.remove_stopwords<a class="headerlink" href="#texthero-preprocessing-remove-stopwords" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.remove_stopwords">
<code class="sig-name descname">remove_stopwords</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">stopwords</span><span class="p">:</span> <span class="n">Union<span class="p">[</span>Set<span class="p">[</span><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.8)">str</a><span class="p">]</span><span class="p">, </span>NoneType<span class="p">]</span></span> <span class="o">=</span> <span class="default_value">None</span></em>, <em class="sig-param"><span class="n">remove_str_numbers</span><span class="o">=</span><span class="default_value">False</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.remove_stopwords" title="Permalink to this definition">¶</a></dt>
<dd><p>Remove all instances of <cite>words</cite>.</p>
<p>By default uses NLTK’s english stopwords of 179 words:</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><dl class="simple">
<dt><strong>input</strong><span class="classifier">Pandas Series</span></dt><dd></dd>
<dt><strong>stopwords</strong><span class="classifier">Set[str], Optional</span></dt><dd><p>Set of stopwords string to remove. If not passed, by default it used NLTK English stopwords.</p>
</dd>
</dl>
</dd>
</dl>
<p class="rubric">Examples</p>
<p>Using default NLTK list of stopwords:</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"Texthero is not only for the heroes"</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">remove_stopwords</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    Texthero      heroes</span>
<span class="go">dtype: object</span>
</pre></div>
</div>
<p>Add custom words into the default list of stopwords:</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">texthero</span> <span class="kn">import</span> <span class="n">stopwords</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">default_stopwords</span> <span class="o">=</span> <span class="n">stopwords</span><span class="o">.</span><span class="n">DEFAULT</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">custom_stopwords</span> <span class="o">=</span> <span class="n">default_stopwords</span><span class="o">.</span><span class="n">union</span><span class="p">(</span><span class="nb">set</span><span class="p">([</span><span class="s2">"heroes"</span><span class="p">]))</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"Texthero is not only for the heroes"</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">remove_stopwords</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="n">custom_stopwords</span><span class="p">)</span>
<span class="go">0    Texthero      </span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>