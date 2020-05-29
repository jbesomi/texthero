---
id: texthero.preprocessing.tokenize
title: preprocessing.tokenize
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-tokenize">
<h1>texthero.preprocessing.tokenize<a class="headerlink" href="#texthero-preprocessing-tokenize" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.tokenize">
<code class="sig-name descname">tokenize</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.tokenize" title="Permalink to this definition">¶</a></dt>
<dd><p>Tokenize each row of the given Series.</p>
<p>Algorithm: add a space closer to a punctuation symbol at
exception if the symbol is between two alphanumeric character and split.</p>
</dd></dl>
</div>
</div>