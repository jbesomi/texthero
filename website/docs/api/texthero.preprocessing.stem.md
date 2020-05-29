---
id: texthero.preprocessing.stem
title: preprocessing.stem
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-stem">
<h1>texthero.preprocessing.stem<a class="headerlink" href="#texthero-preprocessing-stem" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.stem">
<code class="sig-name descname">stem</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">stem</span><span class="o">=</span><span class="default_value">'snowball'</span></em>, <em class="sig-param"><span class="n">language</span><span class="o">=</span><span class="default_value">'english'</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.stem" title="Permalink to this definition">¶</a></dt>
<dd><p>Stem series using either ‘porter’ or ‘snowball’ NLTK stemmers.</p>
<p>Not in the default pipeline.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><dl class="simple">
<dt><strong>input</strong></dt><dd></dd>
<dt><strong>stem</strong></dt><dd><p>Can be either ‘snowball’ or ‘porter’. (“snowball” is default)</p>
</dd>
<dt><strong>language</strong></dt><dd><dl class="simple">
<dt>Supported languages: </dt><dd><p>danish dutch english finnish french german hungarian italian 
norwegian portuguese romanian russian spanish swedish</p>
</dd>
</dl>
</dd>
</dl>
</dd>
</dl>
</dd></dl>
</div>
</div>