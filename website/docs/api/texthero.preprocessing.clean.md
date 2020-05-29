---
id: texthero.preprocessing.clean
title: preprocessing.clean
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-clean">
<h1>texthero.preprocessing.clean<a class="headerlink" href="#texthero-preprocessing-clean" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.clean">
<code class="sig-name descname">clean</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">pipeline</span><span class="o">=</span><span class="default_value">None</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.clean" title="Permalink to this definition">¶</a></dt>
<dd><p>Clean pandas series by appling a preprocessing pipeline.</p>
<p>For information regarding a specific function type <cite>help(texthero.preprocessing.func_name)</cite>.</p>
<p>The default preprocessing pipeline is the following:
- fillna
- lowercase
- remove_digits
- remove_punctuation
- remove_diacritics
- remove_stopwords
- remove_whitespace</p>
</dd></dl>
</div>
</div>