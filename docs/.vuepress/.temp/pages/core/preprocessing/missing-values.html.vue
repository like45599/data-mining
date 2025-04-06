<template><div><h1 id="缺失值处理" tabindex="-1"><a class="header-anchor" href="#缺失值处理"><span>缺失值处理</span></a></h1>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>了解缺失值产生的原因及其影响</li>
      <li>掌握缺失值检测与分析方法</li>
      <li>学习多种缺失值处理策略及其适用场景</li>
      <li>实践常用的缺失值处理技术</li>
    </ul>
  </div>
</div>
<h2 id="缺失值概述" tabindex="-1"><a class="header-anchor" href="#缺失值概述"><span>缺失值概述</span></a></h2>
<p>缺失值是数据分析和机器学习中常见的问题，有效处理缺失值对模型性能至关重要。</p>
<h3 id="缺失值产生的原因" tabindex="-1"><a class="header-anchor" href="#缺失值产生的原因"><span>缺失值产生的原因</span></a></h3>
<p>缺失值通常由以下原因导致：</p>
<ol>
<li><strong>数据收集问题</strong>：如调查问卷未回答、传感器故障</li>
<li><strong>数据整合问题</strong>：合并多个来源的数据时信息不完整</li>
<li><strong>数据处理错误</strong>：导入、转换或清洗过程中的错误</li>
<li><strong>隐私保护</strong>：有意隐藏某些敏感信息</li>
<li><strong>结构性缺失</strong>：某些条件下不需要收集的数据</li>
</ol>
<h3 id="缺失值的类型" tabindex="-1"><a class="header-anchor" href="#缺失值的类型"><span>缺失值的类型</span></a></h3>
<p>根据缺失机制，缺失值可分为三类：</p>
<ol>
<li>
<p><strong>完全随机缺失(MCAR, Missing Completely At Random)</strong></p>
<ul>
<li>缺失完全随机，与任何观测或未观测变量无关</li>
<li>例如：实验设备随机故障导致的数据丢失</li>
</ul>
</li>
<li>
<p><strong>随机缺失(MAR, Missing At Random)</strong></p>
<ul>
<li>缺失概率只与观测到的其他变量有关</li>
<li>例如：年龄较大的人更可能不回答收入问题</li>
</ul>
</li>
<li>
<p><strong>非随机缺失(MNAR, Missing Not At Random)</strong></p>
<ul>
<li>缺失概率与未观测到的变量或缺失值本身有关</li>
<li>例如：高收入人群不愿透露收入</li>
</ul>
</li>
</ol>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>统计学家唐纳德·鲁宾(Donald Rubin)于1976年首次系统地提出了缺失数据机制的分类(MCAR、MAR、MNAR)，这一理论框架至今仍是处理缺失值的基础。不同类型的缺失机制需要不同的处理策略，选择合适的方法对分析结果的可靠性至关重要。</p>
  </div>
</div>
<h2 id="缺失值分析" tabindex="-1"><a class="header-anchor" href="#缺失值分析"><span>缺失值分析</span></a></h2>
<h3 id="_1-检测缺失值" tabindex="-1"><a class="header-anchor" href="#_1-检测缺失值"><span>1. 检测缺失值</span></a></h3>
<p>在处理缺失值前，首先需要全面了解数据中缺失值的情况：</p>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd</span>
<span class="line"><span class="token keyword">import</span> numpy <span class="token keyword">as</span> np</span>
<span class="line"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt</span>
<span class="line"><span class="token keyword">import</span> seaborn <span class="token keyword">as</span> sns</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 加载数据</span></span>
<span class="line">df <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'data.csv'</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 检查每列的缺失值数量</span></span>
<span class="line">missing_values <span class="token operator">=</span> df<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>missing_values<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 计算缺失比例</span></span>
<span class="line">missing_ratio <span class="token operator">=</span> df<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">/</span> <span class="token builtin">len</span><span class="token punctuation">(</span>df<span class="token punctuation">)</span> <span class="token operator">*</span> <span class="token number">100</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>missing_ratio<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 可视化缺失值</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">missing_ratio <span class="token operator">=</span> missing_ratio<span class="token punctuation">[</span>missing_ratio <span class="token operator">></span> <span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">.</span>sort_values<span class="token punctuation">(</span>ascending<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>barplot<span class="token punctuation">(</span>x<span class="token operator">=</span>missing_ratio<span class="token punctuation">.</span>index<span class="token punctuation">,</span> y<span class="token operator">=</span>missing_ratio<span class="token punctuation">.</span>values<span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'缺失值比例'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xticks<span class="token punctuation">(</span>rotation<span class="token operator">=</span><span class="token number">45</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'缺失百分比'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h3 id="_2-缺失模式分析" tabindex="-1"><a class="header-anchor" href="#_2-缺失模式分析"><span>2. 缺失模式分析</span></a></h3>
<p>理解缺失值之间的关系有助于选择合适的处理策略：</p>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># 缺失值热图</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">8</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>heatmap<span class="token punctuation">(</span>df<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> cbar<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">,</span> cmap<span class="token operator">=</span><span class="token string">'viridis'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'缺失值分布热图'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 缺失值相关性</span></span>
<span class="line">missing_binary <span class="token operator">=</span> df<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>astype<span class="token punctuation">(</span><span class="token builtin">int</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">8</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>heatmap<span class="token punctuation">(</span>missing_binary<span class="token punctuation">.</span>corr<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> annot<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span> cmap<span class="token operator">=</span><span class="token string">'coolwarm'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'缺失值相关性热图'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<div class="visualization-container">
  <div class="visualization-title">缺失值模式可视化</div>
  <div class="visualization-content">
    <img src="/images/missing_pattern_en.svg" alt="缺失值模式可视化">
  </div>
  <div class="visualization-caption">
    图: 缺失值分布热图。黄色区域表示缺失值，可以观察到缺失模式的分布情况。
  </div>
</div>
<h2 id="缺失值处理策略" tabindex="-1"><a class="header-anchor" href="#缺失值处理策略"><span>缺失值处理策略</span></a></h2>
<h3 id="_1-删除含缺失值的数据" tabindex="-1"><a class="header-anchor" href="#_1-删除含缺失值的数据"><span>1. 删除含缺失值的数据</span></a></h3>
<p>最简单的处理方法，但可能导致信息丢失：</p>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># 删除所有含缺失值的行</span></span>
<span class="line">df_dropped_rows <span class="token operator">=</span> df<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 删除缺失比例超过50%的列</span></span>
<span class="line">threshold <span class="token operator">=</span> <span class="token builtin">len</span><span class="token punctuation">(</span>df<span class="token punctuation">)</span> <span class="token operator">*</span> <span class="token number">0.5</span></span>
<span class="line">df_dropped_cols <span class="token operator">=</span> df<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span>axis<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">,</span> thresh<span class="token operator">=</span>threshold<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 只删除特定列缺失的行</span></span>
<span class="line">df_dropped_specific <span class="token operator">=</span> df<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span>subset<span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">'income'</span><span class="token punctuation">,</span> <span class="token string">'age'</span><span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<p><strong>适用场景</strong>：</p>
<ul>
<li>缺失值比例很小（&lt;5%）</li>
<li>数据量充足，删除不会显著减少样本量</li>
<li>缺失值完全随机(MCAR)</li>
</ul>
<p><strong>缺点</strong>：</p>
<ul>
<li>可能引入偏差，特别是当缺失不是完全随机时</li>
<li>减少样本量，降低统计功效</li>
<li>可能丢失重要信息</li>
</ul>
<h3 id="_2-填充缺失值" tabindex="-1"><a class="header-anchor" href="#_2-填充缺失值"><span>2. 填充缺失值</span></a></h3>
<h4 id="_2-1-统计量填充" tabindex="-1"><a class="header-anchor" href="#_2-1-统计量填充"><span>2.1 统计量填充</span></a></h4>
<p>使用统计量代替缺失值：</p>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># 均值填充</span></span>
<span class="line">df_mean <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">df_mean<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_mean<span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> inplace<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 中位数填充</span></span>
<span class="line">df_median <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">df_median<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_median<span class="token punctuation">.</span>median<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> inplace<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 众数填充</span></span>
<span class="line">df_mode <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">for</span> col <span class="token keyword">in</span> df_mode<span class="token punctuation">.</span>columns<span class="token punctuation">:</span></span>
<span class="line">    <span class="token keyword">if</span> df_mode<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>dtype <span class="token operator">==</span> <span class="token string">'object'</span><span class="token punctuation">:</span>  <span class="token comment"># 类别变量</span></span>
<span class="line">        df_mode<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_mode<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>mode<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> inplace<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<div class="interactive-component">
  <div class="interactive-title">填充方法比较</div>
  <div class="interactive-content">
    <missing-value-imputation></missing-value-imputation>
  </div>
  <div class="interactive-caption">
    交互组件：尝试不同的填充方法，观察对数据分布的影响。
  </div>
</div>
<h4 id="_2-2-高级填充方法" tabindex="-1"><a class="header-anchor" href="#_2-2-高级填充方法"><span>2.2 高级填充方法</span></a></h4>
<p>利用数据间的关系进行更智能的填充：</p>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>impute <span class="token keyword">import</span> KNNImputer</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>experimental <span class="token keyword">import</span> enable_iterative_imputer</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>impute <span class="token keyword">import</span> IterativeImputer</span>
<span class="line"></span>
<span class="line"><span class="token comment"># KNN填充</span></span>
<span class="line">imputer <span class="token operator">=</span> KNNImputer<span class="token punctuation">(</span>n_neighbors<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">)</span></span>
<span class="line">df_knn <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span></span>
<span class="line">    imputer<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df<span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    columns<span class="token operator">=</span>df<span class="token punctuation">.</span>columns</span>
<span class="line"><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 多重插补（使用MICE算法的简化版本）</span></span>
<span class="line">imputer <span class="token operator">=</span> IterativeImputer<span class="token punctuation">(</span>max_iter<span class="token operator">=</span><span class="token number">10</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">df_mice <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span></span>
<span class="line">    imputer<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df<span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    columns<span class="token operator">=</span>df<span class="token punctuation">.</span>columns</span>
<span class="line"><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h2 id="缺失值处理评估" tabindex="-1"><a class="header-anchor" href="#缺失值处理评估"><span>缺失值处理评估</span></a></h2>
<h3 id="_1-比较不同方法的效果" tabindex="-1"><a class="header-anchor" href="#_1-比较不同方法的效果"><span>1. 比较不同方法的效果</span></a></h3>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt</span>
<span class="line"><span class="token keyword">import</span> seaborn <span class="token keyword">as</span> sns</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 原始数据分布</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">15</span><span class="token punctuation">,</span> <span class="token number">5</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>subplot<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>histplot<span class="token punctuation">(</span>df<span class="token punctuation">[</span><span class="token string">'income'</span><span class="token punctuation">]</span><span class="token punctuation">.</span>dropna<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> kde<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'原始数据分布'</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 均值填充后的分布</span></span>
<span class="line">plt<span class="token punctuation">.</span>subplot<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>histplot<span class="token punctuation">(</span>df_mean<span class="token punctuation">[</span><span class="token string">'income'</span><span class="token punctuation">]</span><span class="token punctuation">,</span> kde<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'均值填充后的分布'</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># KNN填充后的分布</span></span>
<span class="line">plt<span class="token punctuation">.</span>subplot<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>histplot<span class="token punctuation">(</span>df_knn<span class="token punctuation">[</span><span class="token string">'income'</span><span class="token punctuation">]</span><span class="token punctuation">,</span> kde<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'KNN填充后的分布'</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">plt<span class="token punctuation">.</span>tight_layout<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h2 id="实践建议" tabindex="-1"><a class="header-anchor" href="#实践建议"><span>实践建议</span></a></h2>
<h3 id="_1-缺失值处理流程" tabindex="-1"><a class="header-anchor" href="#_1-缺失值处理流程"><span>1. 缺失值处理流程</span></a></h3>
<ol>
<li>
<p><strong>探索性分析</strong>：</p>
<ul>
<li>检测缺失值的位置、数量和比例</li>
<li>分析缺失模式和机制</li>
<li>可视化缺失值分布</li>
</ul>
</li>
<li>
<p><strong>制定处理策略</strong>：</p>
<ul>
<li>根据缺失机制选择合适的处理方法</li>
<li>考虑特征的重要性和数据结构</li>
<li>可能需要不同特征采用不同策略</li>
</ul>
</li>
<li>
<p><strong>实施与评估</strong>：</p>
<ul>
<li>实施选定的填充方法</li>
<li>比较不同方法的效果</li>
<li>验证处理后数据的质量和一致性</li>
</ul>
</li>
</ol>
<h3 id="_2-实际应用技巧" tabindex="-1"><a class="header-anchor" href="#_2-实际应用技巧"><span>2. 实际应用技巧</span></a></h3>
<ul>
<li><strong>先分析后处理</strong>：深入了解缺失原因再选择方法</li>
<li><strong>特征相关性</strong>：利用特征间关系改进填充效果</li>
<li><strong>领域知识</strong>：结合业务理解指导缺失值处理</li>
<li><strong>敏感性分析</strong>：测试不同填充方法对最终结果的影响</li>
<li><strong>保留不确定性</strong>：考虑使用多重插补保留估计的不确定性</li>
</ul>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见误区
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>盲目删除</strong>：不分析缺失机制就直接删除含缺失值的样本</li>
      <li><strong>过度依赖均值</strong>：对所有特征都使用均值填充，忽略数据分布特性</li>
      <li><strong>忽略缺失值相关性</strong>：未考虑特征间关系进行填充</li>
      <li><strong>数据泄露</strong>：使用测试集信息填充训练集缺失值</li>
    </ul>
  </div>
</div>
<h2 id="小结与思考" tabindex="-1"><a class="header-anchor" href="#小结与思考"><span>小结与思考</span></a></h2>
<p>缺失值处理是数据预处理中的关键步骤，合适的处理方法可以提高模型性能并确保分析结果的可靠性。</p>
<h3 id="关键要点回顾" tabindex="-1"><a class="header-anchor" href="#关键要点回顾"><span>关键要点回顾</span></a></h3>
<ul>
<li>缺失值可能由多种原因导致，包括数据收集问题、隐私保护等</li>
<li>缺失机制分为MCAR、MAR和MNAR三种类型</li>
<li>处理策略包括删除法和多种填充方法</li>
<li>选择合适的处理方法需要考虑缺失机制、数据特性和分析目标</li>
</ul>
<h3 id="思考问题" tabindex="-1"><a class="header-anchor" href="#思考问题"><span>思考问题</span></a></h3>
<ol>
<li>如何判断数据中的缺失机制类型？</li>
<li>在什么情况下，删除含缺失值的样本比填充更合适？</li>
<li>如何评估缺失值处理方法的有效性？</li>
</ol>
<BackToPath /><div class="practice-link">
  <a href="/projects/preprocessing.html" class="button">前往实践项目</a>
</div> </div></template>


