---
id: texthero.representation.tfidf
title: representation.tfidf
hide_title: true
---

<div>
<div class="section" id="texthero-representation-tfidf">
<h1>texthero.representation.tfidf<a class="headerlink" href="#texthero-representation-tfidf" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.representation.tfidf">
<code class="sig-name descname">tfidf</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">max_features</span><span class="o">=</span><span class="default_value">None</span></em>, <em class="sig-param"><span class="n">min_df</span><span class="o">=</span><span class="default_value">1</span></em>, <em class="sig-param"><span class="n">return_feature_names</span><span class="o">=</span><span class="default_value">False</span></em><span class="sig-paren">)</span><a class="headerlink" href="#texthero.representation.tfidf" title="Permalink to this definition">¶</a></dt>
<dd><p>Represent a text-based Pandas Series using TF-IDF.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><dl class="simple">
<dt><strong>s</strong><span class="classifier">Pandas Series</span></dt><dd></dd>
<dt><strong>max_features</strong><span class="classifier">int, optional</span></dt><dd><p>Maximum number of features to keep.</p>
</dd>
<dt><strong>min_df</strong><span class="classifier">int, optional. Default to 1.</span></dt><dd><p>When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.</p>
</dd>
<dt><strong>return_features_names</strong><span class="classifier">Boolean. Default to False.</span></dt><dd><p>If True, return a tuple (<em>tfidf_series</em>, <em>features_names</em>)</p>
</dd>
</dl>
</dd>
</dl>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">([</span><span class="s2">"Sentence one"</span><span class="p">,</span> <span class="s2">"Sentence two"</span><span class="p">])</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">tfidf</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    [0.5797386715376657, 0.8148024746671689, 0.0]</span>
<span class="go">1    [0.5797386715376657, 0.0, 0.8148024746671689]</span>
<span class="go">dtype: object</span>
</pre></div>
</div>
<p>To return the <em>feature_names</em>:</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">([</span><span class="s2">"Sentence one"</span><span class="p">,</span> <span class="s2">"Sentence two"</span><span class="p">])</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">tfidf</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="n">return_feature_names</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="go">(0    [0.5797386715376657, 0.8148024746671689, 0.0]</span>
<span class="go">1    [0.5797386715376657, 0.0, 0.8148024746671689]</span>
<span class="go">dtype: object, ['Sentence', 'one', 'two'])</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>