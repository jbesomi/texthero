---
id: texthero.representation.term_frequency
title: representation.term_frequency
hide_title: true
---

<div>
<div class="section" id="texthero-representation-term-frequency">
<h1>texthero.representation.term_frequency<a class="headerlink" href="#texthero-representation-term-frequency" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.representation.term_frequency">
<code class="sig-name descname">term_frequency</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">max_features</span><span class="p">:</span> <span class="n">Union<span class="p">[</span><a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.8)">int</a><span class="p">, </span>NoneType<span class="p">]</span></span> <span class="o">=</span> <span class="default_value">None</span></em>, <em class="sig-param"><span class="n">return_feature_names</span><span class="o">=</span><span class="default_value">False</span></em><span class="sig-paren">)</span><a class="headerlink" href="#texthero.representation.term_frequency" title="Permalink to this definition">¶</a></dt>
<dd><p>Represent a text-based Pandas Series using term_frequency.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><dl class="simple">
<dt><strong>s</strong><span class="classifier">Pandas Series</span></dt><dd></dd>
<dt><strong>max_features</strong><span class="classifier">int, optional</span></dt><dd><p>Maximum number of features to keep.</p>
</dd>
<dt><strong>return_features_names</strong><span class="classifier">Boolean, False by Default</span></dt><dd><p>If True, return a tuple (<em>term_frequency_series</em>, <em>features_names</em>)</p>
</dd>
</dl>
</dd>
</dl>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">([</span><span class="s2">"Sentence one"</span><span class="p">,</span> <span class="s2">"Sentence two"</span><span class="p">])</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">term_frequency</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    [1, 1, 0]</span>
<span class="go">1    [1, 0, 1]</span>
<span class="go">dtype: object</span>
</pre></div>
</div>
<p>To return the features_names:</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">([</span><span class="s2">"Sentence one"</span><span class="p">,</span> <span class="s2">"Sentence two"</span><span class="p">])</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">term_frequency</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="n">return_feature_names</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="go">(0    [1, 1, 0]</span>
<span class="go">1    [1, 0, 1]</span>
<span class="go">dtype: object, ['Sentence', 'one', 'two'])</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>