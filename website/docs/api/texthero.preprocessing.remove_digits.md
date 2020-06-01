---
id: texthero.preprocessing.remove_digits
title: preprocessing.remove_digits
hide_title: true
---

<div>
<div class="section" id="texthero-preprocessing-remove-digits">
<h1>texthero.preprocessing.remove_digits<a class="headerlink" href="#texthero-preprocessing-remove-digits" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.preprocessing.remove_digits">
<code class="sig-name descname">remove_digits</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input</span><span class="p">:</span> <span class="n">pandas.core.series.Series</span></em>, <em class="sig-param"><span class="n">only_blocks</span><span class="o">=</span><span class="default_value">True</span></em><span class="sig-paren">)</span> → pandas.core.series.Series<a class="headerlink" href="#texthero.preprocessing.remove_digits" title="Permalink to this definition">¶</a></dt>
<dd><p>Remove all digits and replace it with a single space.</p>
<p>By default, only removes “blocks” of digits. For instance, <cite>1234 falcon9</cite> becomes ` falcon9`.</p>
<p>When the arguments <cite>only_blocks</cite> is set to ´False´, remove any digits.</p>
<p>See also <code class="xref py py-meth docutils literal notranslate"><span class="pre">replace_digits()</span></code> to replace digits with another string.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><dl class="simple">
<dt><strong>input</strong><span class="classifier">Pandas Series</span></dt><dd></dd>
<dt><strong>only_blocks</strong><span class="classifier">bool</span></dt><dd><p>Remove only blocks of digits.</p>
</dd>
</dl>
</dd>
</dl>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"7ex7hero is fun 1111"</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">preprocessing</span><span class="o">.</span><span class="n">remove_digits</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    7ex7hero is fun  </span>
<span class="go">dtype: object</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">preprocessing</span><span class="o">.</span><span class="n">remove_digits</span><span class="p">(</span><span class="n">s</span><span class="p">,</span> <span class="n">only_blocks</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="go">0     ex hero is fun  </span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>