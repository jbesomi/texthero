---
id: texthero.nlp.named_entities
title: nlp.named_entities
hide_title: true
---

<div>
<div class="section" id="texthero-nlp-named-entities">
<h1>texthero.nlp.named_entities<a class="headerlink" href="#texthero-nlp-named-entities" title="Permalink to this headline">¶</a></h1>
<dl class="py function">
<dt id="texthero.nlp.named_entities">
<code class="sig-name descname">named_entities</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">s</span></em>, <em class="sig-param"><span class="n">package</span><span class="o">=</span><span class="default_value">'spacy'</span></em><span class="sig-paren">)</span><a class="headerlink" href="#texthero.nlp.named_entities" title="Permalink to this definition">¶</a></dt>
<dd><p>Return named-entities.</p>
<p>Return a Pandas Series where each rows contains a list of tuples containing information regarding the given named entities.</p>
<p>Tuple: (<cite>entity’name</cite>, <cite>entity’label</cite>, <cite>starting character</cite>, <cite>ending character</cite>)</p>
<p>Under the hood, <cite>named_entities</cite> make use of Spacy name entity recognition.</p>
<dl class="simple">
<dt>List of labels:</dt><dd><ul class="simple">
<li><p><cite>PERSON</cite>: People, including fictional.</p></li>
<li><p><cite>NORP</cite>: Nationalities or religious or political groups.</p></li>
<li><p><cite>FAC</cite>: Buildings, airports, highways, bridges, etc.</p></li>
<li><p><cite>ORG</cite> : Companies, agencies, institutions, etc.</p></li>
<li><p><cite>GPE</cite>: Countries, cities, states.</p></li>
<li><p><cite>LOC</cite>: Non-GPE locations, mountain ranges, bodies of water.</p></li>
<li><p><cite>PRODUCT</cite>: Objects, vehicles, foods, etc. (Not services.)</p></li>
<li><p><cite>EVENT</cite>: Named hurricanes, battles, wars, sports events, etc.</p></li>
<li><p><cite>WORK_OF_ART</cite>: Titles of books, songs, etc.</p></li>
<li><p><cite>LAW</cite>: Named documents made into laws.</p></li>
<li><p><cite>LANGUAGE</cite>: Any named language.</p></li>
<li><p><cite>DATE</cite>: Absolute or relative dates or periods.</p></li>
<li><p><cite>TIME</cite>: Times smaller than a day.</p></li>
<li><p><cite>PERCENT</cite>: Percentage, including ”%“.</p></li>
<li><p><cite>MONEY</cite>: Monetary values, including unit.</p></li>
<li><p><cite>QUANTITY</cite>: Measurements, as of weight or distance.</p></li>
<li><p><cite>ORDINAL</cite>: “first”, “second”, etc.</p></li>
<li><p><cite>CARDINAL</cite>: Numerals that do not fall under another type.</p></li>
</ul>
</dd>
</dl>
<p class="rubric">Examples</p>
<div class="doctest highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">texthero</span> <span class="k">as</span> <span class="nn">hero</span>
<span class="gp">&gt;&gt;&gt; </span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">s</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="s2">"Yesterday I was in NY with Bill de Blasio"</span><span class="p">)</span>
<span class="gp">&gt;&gt;&gt; </span><span class="n">hero</span><span class="o">.</span><span class="n">named_entities</span><span class="p">(</span><span class="n">s</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
<span class="go">[('Yesterday', 'DATE', 0, 9), ('NY', 'GPE', 19, 21), ('Bill de Blasio', 'PERSON', 27, 41)]</span>
</pre></div>
</div>
</dd></dl>
</div>
</div>