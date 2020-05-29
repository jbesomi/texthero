---
id: texthero.visualization.top_words
title: visualization.top_words
hide_title: true
---

<div>
<div class="section" id="texthero-visualization-top-words">
<h1>texthero.visualization.top_words<a class="headerlink" href="#texthero-visualization-top-words" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.visualization.top_words">
<code class="sig-name descname">top_words</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">normalize</span><span class="o">=</span><span class="default_value">False</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.visualization.top_words" title="Permalink to this definition">¶</a></dt>
<dd><p>Return a pandas series with index the top words and as value the count.</p>
<p>Tokenization: split by space and remove all punctuations that are not between characters.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><dl class="simple">
<dt><strong>normalize :</strong></dt><dd><p>When set to true, return normalized values.</p>
</dd>
</dl>
</dd>
</dl>
</dd></dl>
</div>
</div>