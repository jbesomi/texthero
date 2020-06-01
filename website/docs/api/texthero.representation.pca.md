---
id: texthero.representation.pca
title: representation.pca
hide_title: true
---

<div>
<div class="section" id="texthero-representation-pca">
<h1>texthero.representation.pca<a class="headerlink" href="#texthero-representation-pca" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.representation.pca">
<code class="sig-name descname">pca</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span></em>, <em class="sig-param"><span class="n">n_components</span><span class="o">=</span><span class="default_value">2</span></em><span class="sig-paren">)</span><a class="headerlink" href="#texthero.representation.pca" title="Permalink to this definition">¶</a></dt>
<dd><p>Perform principal component analysis on the given Pandas Series.</p>
<p>In general, <em>pca</em> should be called after the text has already been represented.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><dl class="simple">
<dt><strong>s</strong><span class="classifier">Pandas Series</span></dt><dd></dd>
<dt><strong>n_components</strong><span class="classifier">Int. Default is 2.</span></dt><dd><p>Number of components to keep. If n_components is not set or None, all components are kept.</p>
</dd>
</dl>
</dd>
</dl>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">([</span><span class="s2">"Sentence one"</span><span class="p">,</span> <span class="s2">"Sentence two"</span><span class="p">])</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>