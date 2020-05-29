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
<code class="sig-name descname">remove_stopwords</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">stopwords</span><span class="p">:</span> <span class="n">Union<span class="p">[</span>Set<span class="p">[</span>str<span class="p">]</span><span class="p">, </span>NoneType<span class="p">]</span></span> <span class="o">=</span> <span class="default_value">None</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.remove_stopwords" title="Permalink to this definition">¶</a></dt>
<dd><p>Remove all instances of <cite>words</cite> and replace it with an empty space.</p>
<p>By default uses NLTK’s english stopwords of 179 words.</p>
</dd></dl>
</div>
</div>