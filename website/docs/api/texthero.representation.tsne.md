---
id: texthero.representation.tsne
title: representation.tsne
hide_title: true
---

<div>
<div class="section" id="texthero-representation-tsne">
<h1>texthero.representation.tsne<a class="headerlink" href="#texthero-representation-tsne" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.representation.tsne">
<code class="sig-name descname">tsne</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">n_components</span><span class="o">=</span><span class="default_value">2</span></em>, <em class="sig-param"><span class="n">perplexity</span><span class="o">=</span><span class="default_value">30.0</span></em>, <em class="sig-param"><span class="n">early_exaggeration</span><span class="o">=</span><span class="default_value">12.0</span></em>, <em class="sig-param"><span class="n">learning_rate</span><span class="o">=</span><span class="default_value">200.0</span></em>, <em class="sig-param"><span class="n">n_iter</span><span class="o">=</span><span class="default_value">1000</span></em>, <em class="sig-param"><span class="n">n_iter_without_progress</span><span class="o">=</span><span class="default_value">300</span></em>, <em class="sig-param"><span class="n">min_grad_norm</span><span class="o">=</span><span class="default_value">1e-07</span></em>, <em class="sig-param"><span class="n">metric</span><span class="o">=</span><span class="default_value">'euclidean'</span></em>, <em class="sig-param"><span class="n">init</span><span class="o">=</span><span class="default_value">'random'</span></em>, <em class="sig-param"><span class="n">verbose</span><span class="o">=</span><span class="default_value">0</span></em>, <em class="sig-param"><span class="n">random_state</span><span class="o">=</span><span class="default_value">None</span></em>, <em class="sig-param"><span class="n">method</span><span class="o">=</span><span class="default_value">'barnes_hut'</span></em>, <em class="sig-param"><span class="n">angle</span><span class="o">=</span><span class="default_value">0.5</span></em>, <em class="sig-param"><span class="n">n_jobs</span><span class="o">=</span><span class="default_value">- 1</span></em><span class="sig-paren">)</span><a class="headerlink" href="#texthero.representation.tsne" title="Permalink to this definition">¶</a></dt>
<dd><p>Perform TSNE on the given pandas series.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><dl class="simple">
<dt><strong>s</strong><span class="classifier">Pandas Series</span></dt><dd></dd>
<dt><strong>n_components</strong><span class="classifier">int, default is 2.</span></dt><dd><p>Number of components to keep. If n_components is not set or None, all components are kept.</p>
</dd>
<dt><strong>perplexity</strong><span class="classifier">int, default is 30.0</span></dt><dd></dd>
</dl>
</dd>
</dl>
</dd></dl>
</div>
</div>