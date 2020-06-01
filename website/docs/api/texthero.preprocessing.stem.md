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
<dd><p>Stem series using either <cite>porter</cite> or <cite>snowball</cite> NLTK stemmers.</p>
<p>The act of stemming means removing the end of a words with an heuristic process. It’s useful in context where the meaning of the word is important rather than his derivation. Stemming is very efficient and adapt in case the given dataset is large.</p>
<p><cite>texthero.preprocessing.stem</cite> make use of two NLTK stemming algorithms known as <code class="xref py py-class docutils literal notranslate"><span class="pre">nltk.stem.SnowballStemmer</span></code> and <code class="xref py py-class docutils literal notranslate"><span class="pre">nltk.stem.PorterStemmer</span></code>. SnowballStemmer should be used when the Pandas Series contains non-English text has it has multilanguage support.</p>
<dl class="field-list simple">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><dl class="simple">
<dt><strong>input</strong><span class="classifier">Pandas Series</span></dt><dd></dd>
<dt><strong>stem</strong><span class="classifier">str (snowball by default)</span></dt><dd><p>Stemming algorithm. It can be either ‘snowball’ or ‘porter’</p>
</dd>
<dt><strong>language</strong><span class="classifier">str (english by default)</span></dt><dd><p>Supported languages: <cite>danish</cite>, <cite>dutch</cite>, <cite>english</cite>, <cite>finnish</cite>, <cite>french</cite>, <cite>german</cite> , <cite>hungarian</cite>, <cite>italian</cite>, <cite>norwegian</cite>, <cite>portuguese</cite>, <cite>romanian</cite>, <cite>russian</cite>, <cite>spanish</cite> and <cite>swedish</cite>.</p>
</dd>
</dl>
</dd>
</dl>
<p class="rubric">Notes</p>
<p>By default NLTK stemming algorithms lowercase all text.</p>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"I used to go </span><span class="se">\t\n</span><span class="s2"> running."</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">preprocessing</span><span class="o">.</span><span class="n">stem</span><span class="p">(</span><span class="n">s</span><span class="p">)</span>
<span class="go">0    i use to go running.</span>
<span class="go">dtype: object</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>