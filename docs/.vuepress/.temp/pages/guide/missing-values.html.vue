<template><div><h1 id="缺失值处理" tabindex="-1"><a class="header-anchor" href="#缺失值处理"><span>缺失值处理</span></a></h1>
<h2 id="知识要点" tabindex="-1"><a class="header-anchor" href="#知识要点"><span>知识要点</span></a></h2>
<h3 id="理论知识" tabindex="-1"><a class="header-anchor" href="#理论知识"><span>理论知识</span></a></h3>
<ul>
<li>缺失值类型识别</li>
<li>MCAR、MAR、MNAR区别</li>
<li>缺失值处理策略</li>
<li>缺失值影响评估</li>
</ul>
<h3 id="实践要点" tabindex="-1"><a class="header-anchor" href="#实践要点"><span>实践要点</span></a></h3>
<ul>
<li>缺失值可视化分析</li>
<li>删除法实现</li>
<li>填充法应用</li>
<li>高级填充方法</li>
</ul>
<h2 id="缺失值类型" tabindex="-1"><a class="header-anchor" href="#缺失值类型"><span>缺失值类型</span></a></h2>
<h3 id="_1-完全随机缺失-mcar" tabindex="-1"><a class="header-anchor" href="#_1-完全随机缺失-mcar"><span>1. 完全随机缺失（MCAR）</span></a></h3>
<ul>
<li>数据的缺失完全是随机的</li>
<li>缺失与其他变量无关</li>
<li>例如：随机抽样过程中的遗漏</li>
</ul>
<h3 id="_2-随机缺失-mar" tabindex="-1"><a class="header-anchor" href="#_2-随机缺失-mar"><span>2. 随机缺失（MAR）</span></a></h3>
<ul>
<li>数据的缺失与其他观察变量相关</li>
<li>与缺失变量本身的值无关</li>
<li>例如：高收入人群倾向于不填写收入信息</li>
</ul>
<h3 id="_3-非随机缺失-mnar" tabindex="-1"><a class="header-anchor" href="#_3-非随机缺失-mnar"><span>3. 非随机缺失（MNAR）</span></a></h3>
<ul>
<li>数据的缺失与缺失值本身相关</li>
<li>最难处理的缺失类型</li>
<li>例如：成绩差的学生倾向于不填写成绩</li>
</ul>
<h2 id="处理方法" tabindex="-1"><a class="header-anchor" href="#处理方法"><span>处理方法</span></a></h2>
<h3 id="_1-删除法" tabindex="-1"><a class="header-anchor" href="#_1-删除法"><span>1. 删除法</span></a></h3>
<ul>
<li>列删除</li>
<li>行删除</li>
</ul>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line">删除含有缺失值的行</span>
<span class="line">df<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">删除缺失值过多的列</span>
<span class="line">df<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span>axis<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">,</span> thresh<span class="token operator">=</span>df<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token number">0.5</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><ol start="2">
<li>填充法
<ul>
<li>均值填充</li>
<li>中位数填充</li>
<li>众数填充</li>
<li>回归填充</li>
</ul>
</li>
</ol>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># 均值填充</span></span>
<span class="line">df<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df<span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">中位数填充</span>
<span class="line">df<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df<span class="token punctuation">.</span>median<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">前向<span class="token operator">/</span>后向填充</span>
<span class="line">df<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>method<span class="token operator">=</span><span class="token string">'ffill'</span><span class="token punctuation">)</span> <span class="token comment"># 前向填充</span></span>
<span class="line">df<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>method<span class="token operator">=</span><span class="token string">'bfill'</span><span class="token punctuation">)</span> <span class="token comment"># 后向填充</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="_3-高级填充方法" tabindex="-1"><a class="header-anchor" href="#_3-高级填充方法"><span>3. 高级填充方法</span></a></h3>
<ul>
<li><strong>回归填充</strong>：使用其他变量预测缺失值</li>
<li><strong>KNN填充</strong>：基于相似样本进行填充</li>
<li><strong>多重插补</strong>：考虑不确定性的填充方法</li>
</ul>
<h2 id="实践案例" tabindex="-1"><a class="header-anchor" href="#实践案例"><span>实践案例</span></a></h2>
<h3 id="数据探索" tabindex="-1"><a class="header-anchor" href="#数据探索"><span>数据探索</span></a></h3>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line">查看缺失值情况</span>
<span class="line"><span class="token keyword">def</span> <span class="token function">missing_values_analysis</span><span class="token punctuation">(</span>df<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">missing <span class="token operator">=</span> df<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">missing_percent <span class="token operator">=</span> missing <span class="token operator">/</span> <span class="token builtin">len</span><span class="token punctuation">(</span>df<span class="token punctuation">)</span> <span class="token number">100</span></span>
<span class="line"><span class="token keyword">return</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span><span class="token punctuation">{</span></span>
<span class="line"><span class="token string">'Missing Values'</span><span class="token punctuation">:</span> missing<span class="token punctuation">,</span></span>
<span class="line"><span class="token string">'Percentage'</span><span class="token punctuation">:</span> missing_percent</span>
<span class="line"><span class="token punctuation">}</span><span class="token punctuation">)</span><span class="token punctuation">.</span>sort_values<span class="token punctuation">(</span><span class="token string">'Percentage'</span><span class="token punctuation">,</span> ascending<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="处理流程" tabindex="-1"><a class="header-anchor" href="#处理流程"><span>处理流程</span></a></h3>
<ol>
<li>分析缺失值模式</li>
<li>选择合适的处理方法</li>
<li>实施缺失值处理</li>
<li>评估处理效果</li>
</ol>
<h3 id="注意事项" tabindex="-1"><a class="header-anchor" href="#注意事项"><span>注意事项</span></a></h3>
<ul>
<li>了解数据缺失的原因</li>
<li>评估不同处理方法的影响</li>
<li>保留原始数据的副本</li>
<li>记录处理过程和依据</li>
</ul>
</div></template>


