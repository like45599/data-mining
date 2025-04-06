<template><div><h1 id="聚类实际应用案例" tabindex="-1"><a class="header-anchor" href="#聚类实际应用案例"><span>聚类实际应用案例</span></a></h1>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>了解聚类分析在不同领域的实际应用</li>
      <li>掌握从业务问题到聚类方案的转化方法</li>
      <li>学习聚类结果的解释和业务价值挖掘</li>
      <li>理解聚类分析在实际应用中的挑战和解决方案</li>
    </ul>
  </div>
</div>
<h2 id="客户分群案例" tabindex="-1"><a class="header-anchor" href="#客户分群案例"><span>客户分群案例</span></a></h2>
<p>客户分群是聚类分析最常见的应用之一，通过将客户划分为不同群体，企业可以制定针对性的营销策略。</p>
<h3 id="业务背景" tabindex="-1"><a class="header-anchor" href="#业务背景"><span>业务背景</span></a></h3>
<p>某电商平台希望通过分析用户行为数据，将用户划分为不同群体，以便制定差异化的营销策略和个性化推荐。</p>
<h3 id="数据准备" tabindex="-1"><a class="header-anchor" href="#数据准备"><span>数据准备</span></a></h3>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd</span>
<span class="line"><span class="token keyword">import</span> numpy <span class="token keyword">as</span> np</span>
<span class="line"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt</span>
<span class="line"><span class="token keyword">import</span> seaborn <span class="token keyword">as</span> sns</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>preprocessing <span class="token keyword">import</span> StandardScaler</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>cluster <span class="token keyword">import</span> KMeans</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>decomposition <span class="token keyword">import</span> PCA</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 加载数据</span></span>
<span class="line">df <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'customer_data.csv'</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 查看数据</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>df<span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>df<span class="token punctuation">.</span>info<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 特征选择</span></span>
<span class="line">features <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">'recency'</span><span class="token punctuation">,</span> <span class="token string">'frequency'</span><span class="token punctuation">,</span> <span class="token string">'monetary'</span><span class="token punctuation">,</span> <span class="token string">'tenure'</span><span class="token punctuation">,</span> <span class="token string">'age'</span><span class="token punctuation">]</span></span>
<span class="line">X <span class="token operator">=</span> df<span class="token punctuation">[</span>features<span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 处理缺失值</span></span>
<span class="line">X <span class="token operator">=</span> X<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>X<span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 特征缩放</span></span>
<span class="line">scaler <span class="token operator">=</span> StandardScaler<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">X_scaled <span class="token operator">=</span> scaler<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>X<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 查看特征相关性</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">8</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>heatmap<span class="token punctuation">(</span>df<span class="token punctuation">[</span>features<span class="token punctuation">]</span><span class="token punctuation">.</span>corr<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> annot<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span> cmap<span class="token operator">=</span><span class="token string">'coolwarm'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'特征相关性矩阵'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h3 id="确定最佳簇数" tabindex="-1"><a class="header-anchor" href="#确定最佳簇数"><span>确定最佳簇数</span></a></h3>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>metrics <span class="token keyword">import</span> silhouette_score</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 使用肘部法则确定最佳K值</span></span>
<span class="line">wcss <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span></span>
<span class="line">silhouette_scores <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span></span>
<span class="line">K_range <span class="token operator">=</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token number">2</span><span class="token punctuation">,</span> <span class="token number">11</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">for</span> k <span class="token keyword">in</span> K_range<span class="token punctuation">:</span></span>
<span class="line">    kmeans <span class="token operator">=</span> KMeans<span class="token punctuation">(</span>n_clusters<span class="token operator">=</span>k<span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">    kmeans<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_scaled<span class="token punctuation">)</span></span>
<span class="line">    wcss<span class="token punctuation">.</span>append<span class="token punctuation">(</span>kmeans<span class="token punctuation">.</span>inertia_<span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># 计算轮廓系数</span></span>
<span class="line">    labels <span class="token operator">=</span> kmeans<span class="token punctuation">.</span>labels_</span>
<span class="line">    silhouette_scores<span class="token punctuation">.</span>append<span class="token punctuation">(</span>silhouette_score<span class="token punctuation">(</span>X_scaled<span class="token punctuation">,</span> labels<span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 可视化肘部法则</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">5</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>subplot<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>K_range<span class="token punctuation">,</span> wcss<span class="token punctuation">,</span> <span class="token string">'o-'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'簇数量 (K)'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'WCSS'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'肘部法则'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>grid<span class="token punctuation">(</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 可视化轮廓系数</span></span>
<span class="line">plt<span class="token punctuation">.</span>subplot<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>K_range<span class="token punctuation">,</span> silhouette_scores<span class="token punctuation">,</span> <span class="token string">'o-'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'簇数量 (K)'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'轮廓系数'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'轮廓系数法'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>grid<span class="token punctuation">(</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">plt<span class="token punctuation">.</span>tight_layout<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 选择最佳K值</span></span>
<span class="line">best_k <span class="token operator">=</span> <span class="token number">4</span>  <span class="token comment"># 根据上述分析确定</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h3 id="聚类分析" tabindex="-1"><a class="header-anchor" href="#聚类分析"><span>聚类分析</span></a></h3>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># 使用最佳K值进行聚类</span></span>
<span class="line">kmeans <span class="token operator">=</span> KMeans<span class="token punctuation">(</span>n_clusters<span class="token operator">=</span>best_k<span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">df<span class="token punctuation">[</span><span class="token string">'cluster'</span><span class="token punctuation">]</span> <span class="token operator">=</span> kmeans<span class="token punctuation">.</span>fit_predict<span class="token punctuation">(</span>X_scaled<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 降维可视化</span></span>
<span class="line">pca <span class="token operator">=</span> PCA<span class="token punctuation">(</span>n_components<span class="token operator">=</span><span class="token number">2</span><span class="token punctuation">)</span></span>
<span class="line">X_pca <span class="token operator">=</span> pca<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>X_scaled<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 可视化聚类结果</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">8</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">scatter <span class="token operator">=</span> plt<span class="token punctuation">.</span>scatter<span class="token punctuation">(</span>X_pca<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> X_pca<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">,</span> c<span class="token operator">=</span>df<span class="token punctuation">[</span><span class="token string">'cluster'</span><span class="token punctuation">]</span><span class="token punctuation">,</span> cmap<span class="token operator">=</span><span class="token string">'viridis'</span><span class="token punctuation">,</span> s<span class="token operator">=</span><span class="token number">50</span><span class="token punctuation">,</span> alpha<span class="token operator">=</span><span class="token number">0.8</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'客户分群结果 (PCA降维)'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'主成分1'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'主成分2'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>colorbar<span class="token punctuation">(</span>scatter<span class="token punctuation">,</span> label<span class="token operator">=</span><span class="token string">'簇标签'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>grid<span class="token punctuation">(</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 分析各簇的特征</span></span>
<span class="line">cluster_centers <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span>scaler<span class="token punctuation">.</span>inverse_transform<span class="token punctuation">(</span>kmeans<span class="token punctuation">.</span>cluster_centers_<span class="token punctuation">)</span><span class="token punctuation">,</span> columns<span class="token operator">=</span>features<span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"簇中心:"</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>cluster_centers<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 各簇的统计描述</span></span>
<span class="line"><span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>best_k<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"\n簇 </span><span class="token interpolation"><span class="token punctuation">{</span>i<span class="token punctuation">}</span></span><span class="token string"> 的统计描述:"</span></span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token keyword">print</span><span class="token punctuation">(</span>df<span class="token punctuation">[</span>df<span class="token punctuation">[</span><span class="token string">'cluster'</span><span class="token punctuation">]</span> <span class="token operator">==</span> i<span class="token punctuation">]</span><span class="token punctuation">[</span>features<span class="token punctuation">]</span><span class="token punctuation">.</span>describe<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h3 id="业务解释与应用" tabindex="-1"><a class="header-anchor" href="#业务解释与应用"><span>业务解释与应用</span></a></h3>
<p>根据聚类结果，我们可以将客户分为以下几个群体：</p>
<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>客户群体</th>
        <th>特征描述</th>
        <th>营销策略</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>高价值忠诚客户</td>
        <td>
          - 购买频率高<br>
          - 消费金额大<br>
          - 最近有购买<br>
          - 客户年龄较长
        </td>
        <td>
          - VIP会员计划<br>
          - 专属优惠<br>
          - 高端产品推荐<br>
          - 忠诚度奖励
        </td>
      </tr>
      <tr>
        <td>潜力客户</td>
        <td>
          - 购买频率中等<br>
          - 消费金额中等<br>
          - 最近有购买<br>
          - 客户年龄较短
        </td>
        <td>
          - 会员升级激励<br>
          - 交叉销售<br>
          - 个性化推荐<br>
          - 限时优惠
        </td>
      </tr>
      <tr>
        <td>休眠客户</td>
        <td>
          - 购买频率低<br>
          - 消费金额中等<br>
          - 最近无购买<br>
          - 客户年龄较长
        </td>
        <td>
          - 重新激活活动<br>
          - 特别折扣<br>
          - 新产品通知<br>
          - 调查反馈
        </td>
      </tr>
      <tr>
        <td>新客户</td>
        <td>
          - 购买频率低<br>
          - 消费金额低<br>
          - 最近有购买<br>
          - 客户年龄短
        </td>
        <td>
          - 欢迎礼包<br>
          - 入门级产品推荐<br>
          - 自动化营销
        </td>
      </tr>
    </tbody>
  </table>
</div>
<h2 id="异常检测案例" tabindex="-1"><a class="header-anchor" href="#异常检测案例"><span>异常检测案例</span></a></h2>
<p>聚类分析可以用于识别数据中的异常点，这在欺诈检测、网络安全等领域非常有用。</p>
<h3 id="业务背景-1" tabindex="-1"><a class="header-anchor" href="#业务背景-1"><span>业务背景</span></a></h3>
<p>某银行需要从大量交易数据中识别可能的欺诈交易。</p>
<h3 id="数据准备与聚类" tabindex="-1"><a class="header-anchor" href="#数据准备与聚类"><span>数据准备与聚类</span></a></h3>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd</span>
<span class="line"><span class="token keyword">import</span> numpy <span class="token keyword">as</span> np</span>
<span class="line"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>preprocessing <span class="token keyword">import</span> StandardScaler</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>cluster <span class="token keyword">import</span> DBSCAN</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>decomposition <span class="token keyword">import</span> PCA</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 加载交易数据</span></span>
<span class="line">df <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'transactions.csv'</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 特征选择</span></span>
<span class="line">features <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">'amount'</span><span class="token punctuation">,</span> <span class="token string">'time_since_last_transaction'</span><span class="token punctuation">,</span> <span class="token string">'distance_from_home'</span><span class="token punctuation">,</span> <span class="token string">'foreign_transaction'</span><span class="token punctuation">,</span> <span class="token string">'high_risk_merchant'</span><span class="token punctuation">]</span></span>
<span class="line">X <span class="token operator">=</span> df<span class="token punctuation">[</span>features<span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 特征缩放</span></span>
<span class="line">scaler <span class="token operator">=</span> StandardScaler<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">X_scaled <span class="token operator">=</span> scaler<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>X<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 使用DBSCAN进行聚类</span></span>
<span class="line">dbscan <span class="token operator">=</span> DBSCAN<span class="token punctuation">(</span>eps<span class="token operator">=</span><span class="token number">0.5</span><span class="token punctuation">,</span> min_samples<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">)</span></span>
<span class="line">clusters <span class="token operator">=</span> dbscan<span class="token punctuation">.</span>fit_predict<span class="token punctuation">(</span>X_scaled<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 将聚类结果添加到原始数据</span></span>
<span class="line">df<span class="token punctuation">[</span><span class="token string">'cluster'</span><span class="token punctuation">]</span> <span class="token operator">=</span> clusters</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 识别异常点（标签为-1的点）</span></span>
<span class="line">outliers <span class="token operator">=</span> df<span class="token punctuation">[</span>df<span class="token punctuation">[</span><span class="token string">'cluster'</span><span class="token punctuation">]</span> <span class="token operator">==</span> <span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"检测到 </span><span class="token interpolation"><span class="token punctuation">{</span><span class="token builtin">len</span><span class="token punctuation">(</span>outliers<span class="token punctuation">)</span><span class="token punctuation">}</span></span><span class="token string"> 个异常交易"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 降维可视化</span></span>
<span class="line">pca <span class="token operator">=</span> PCA<span class="token punctuation">(</span>n_components<span class="token operator">=</span><span class="token number">2</span><span class="token punctuation">)</span></span>
<span class="line">X_pca <span class="token operator">=</span> pca<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>X_scaled<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">8</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>scatter<span class="token punctuation">(</span>X_pca<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> X_pca<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">,</span> c<span class="token operator">=</span>clusters<span class="token punctuation">,</span> cmap<span class="token operator">=</span><span class="token string">'viridis'</span><span class="token punctuation">,</span> s<span class="token operator">=</span><span class="token number">50</span><span class="token punctuation">,</span> alpha<span class="token operator">=</span><span class="token number">0.8</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>scatter<span class="token punctuation">(</span>X_pca<span class="token punctuation">[</span>clusters <span class="token operator">==</span> <span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> X_pca<span class="token punctuation">[</span>clusters <span class="token operator">==</span> <span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">,</span> c<span class="token operator">=</span><span class="token string">'red'</span><span class="token punctuation">,</span> s<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> alpha<span class="token operator">=</span><span class="token number">0.8</span><span class="token punctuation">,</span> marker<span class="token operator">=</span><span class="token string">'X'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'交易异常检测'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'主成分1'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'主成分2'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>colorbar<span class="token punctuation">(</span>label<span class="token operator">=</span><span class="token string">'簇标签'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 分析异常交易的特征</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"异常交易的特征统计:"</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>outliers<span class="token punctuation">[</span>features<span class="token punctuation">]</span><span class="token punctuation">.</span>describe<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"\n正常交易的特征统计:"</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>df<span class="token punctuation">[</span>df<span class="token punctuation">[</span><span class="token string">'cluster'</span><span class="token punctuation">]</span> <span class="token operator">!=</span> <span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">[</span>features<span class="token punctuation">]</span><span class="token punctuation">.</span>describe<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h3 id="业务建议" tabindex="-1"><a class="header-anchor" href="#业务建议"><span>业务建议</span></a></h3>
<p>基于异常检测结果，可以提出以下建议：</p>
<ol>
<li><strong>实时监控系统</strong>：将聚类模型集成到实时交易监控系统中</li>
<li><strong>风险评分</strong>：为每笔交易计算异常分数，超过阈值时触发人工审核</li>
<li><strong>分层防御</strong>：结合规则引擎和机器学习模型，构建多层欺诈防御系统</li>
<li><strong>持续更新</strong>：定期使用新数据重新训练模型，适应欺诈模式的变化</li>
</ol>
<h2 id="文档聚类案例" tabindex="-1"><a class="header-anchor" href="#文档聚类案例"><span>文档聚类案例</span></a></h2>
<p>聚类分析可以用于组织和分类大量文本文档，帮助信息检索和主题发现。</p>
<h3 id="业务背景-2" tabindex="-1"><a class="header-anchor" href="#业务背景-2"><span>业务背景</span></a></h3>
<p>某新闻网站需要自动对大量新闻文章进行分类，以便更好地组织内容和推荐相关文章。</p>
<h3 id="文本预处理与特征提取" tabindex="-1"><a class="header-anchor" href="#文本预处理与特征提取"><span>文本预处理与特征提取</span></a></h3>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd</span>
<span class="line"><span class="token keyword">import</span> numpy <span class="token keyword">as</span> np</span>
<span class="line"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>feature_extraction<span class="token punctuation">.</span>text <span class="token keyword">import</span> TfidfVectorizer</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>cluster <span class="token keyword">import</span> KMeans</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>decomposition <span class="token keyword">import</span> TruncatedSVD</span>
<span class="line"><span class="token keyword">import</span> nltk</span>
<span class="line"><span class="token keyword">from</span> nltk<span class="token punctuation">.</span>corpus <span class="token keyword">import</span> stopwords</span>
<span class="line"><span class="token keyword">from</span> nltk<span class="token punctuation">.</span>stem <span class="token keyword">import</span> WordNetLemmatizer</span>
<span class="line"><span class="token keyword">import</span> re</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 下载必要的NLTK资源</span></span>
<span class="line">nltk<span class="token punctuation">.</span>download<span class="token punctuation">(</span><span class="token string">'stopwords'</span><span class="token punctuation">)</span></span>
<span class="line">nltk<span class="token punctuation">.</span>download<span class="token punctuation">(</span><span class="token string">'wordnet'</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 加载新闻数据</span></span>
<span class="line">df <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'news_articles.csv'</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 文本预处理函数</span></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">preprocess_text</span><span class="token punctuation">(</span>text<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    <span class="token comment"># 转换为小写</span></span>
<span class="line">    text <span class="token operator">=</span> text<span class="token punctuation">.</span>lower<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token comment"># 移除特殊字符和数字</span></span>
<span class="line">    text <span class="token operator">=</span> re<span class="token punctuation">.</span>sub<span class="token punctuation">(</span><span class="token string">r'[^a-zA-Z\s]'</span><span class="token punctuation">,</span> <span class="token string">''</span><span class="token punctuation">,</span> text<span class="token punctuation">)</span></span>
<span class="line">    <span class="token comment"># 分词</span></span>
<span class="line">    tokens <span class="token operator">=</span> text<span class="token punctuation">.</span>split<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token comment"># 移除停用词</span></span>
<span class="line">    stop_words <span class="token operator">=</span> <span class="token builtin">set</span><span class="token punctuation">(</span>stopwords<span class="token punctuation">.</span>words<span class="token punctuation">(</span><span class="token string">'english'</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">    tokens <span class="token operator">=</span> <span class="token punctuation">[</span>word <span class="token keyword">for</span> word <span class="token keyword">in</span> tokens <span class="token keyword">if</span> word <span class="token keyword">not</span> <span class="token keyword">in</span> stop_words<span class="token punctuation">]</span></span>
<span class="line">    <span class="token comment"># 词形还原</span></span>
<span class="line">    lemmatizer <span class="token operator">=</span> WordNetLemmatizer<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    tokens <span class="token operator">=</span> <span class="token punctuation">[</span>lemmatizer<span class="token punctuation">.</span>lemmatize<span class="token punctuation">(</span>word<span class="token punctuation">)</span> <span class="token keyword">for</span> word <span class="token keyword">in</span> tokens<span class="token punctuation">]</span></span>
<span class="line">    <span class="token comment"># 重新组合为文本</span></span>
<span class="line">    <span class="token keyword">return</span> <span class="token string">' '</span><span class="token punctuation">.</span>join<span class="token punctuation">(</span>tokens<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 应用预处理</span></span>
<span class="line">df<span class="token punctuation">[</span><span class="token string">'processed_text'</span><span class="token punctuation">]</span> <span class="token operator">=</span> df<span class="token punctuation">[</span><span class="token string">'content'</span><span class="token punctuation">]</span><span class="token punctuation">.</span><span class="token builtin">apply</span><span class="token punctuation">(</span>preprocess_text<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 使用TF-IDF提取特征</span></span>
<span class="line">vectorizer <span class="token operator">=</span> TfidfVectorizer<span class="token punctuation">(</span>max_features<span class="token operator">=</span><span class="token number">1000</span><span class="token punctuation">)</span></span>
<span class="line">X <span class="token operator">=</span> vectorizer<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df<span class="token punctuation">[</span><span class="token string">'processed_text'</span><span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 降维以便可视化</span></span>
<span class="line">svd <span class="token operator">=</span> TruncatedSVD<span class="token punctuation">(</span>n_components<span class="token operator">=</span><span class="token number">2</span><span class="token punctuation">)</span></span>
<span class="line">X_svd <span class="token operator">=</span> svd<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>X<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 确定最佳簇数</span></span>
<span class="line">wcss <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span></span>
<span class="line">K_range <span class="token operator">=</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token number">2</span><span class="token punctuation">,</span> <span class="token number">11</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">for</span> k <span class="token keyword">in</span> K_range<span class="token punctuation">:</span></span>
<span class="line">    kmeans <span class="token operator">=</span> KMeans<span class="token punctuation">(</span>n_clusters<span class="token operator">=</span>k<span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">    kmeans<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X<span class="token punctuation">)</span></span>
<span class="line">    wcss<span class="token punctuation">.</span>append<span class="token punctuation">(</span>kmeans<span class="token punctuation">.</span>inertia_<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 可视化肘部法则</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>K_range<span class="token punctuation">,</span> wcss<span class="token punctuation">,</span> <span class="token string">'o-'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'簇数量 (K)'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'WCSS'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'肘部法则'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>grid<span class="token punctuation">(</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 选择最佳K值</span></span>
<span class="line">best_k <span class="token operator">=</span> <span class="token number">5</span>  <span class="token comment"># 根据上述分析确定</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h3 id="聚类分析与主题提取" tabindex="-1"><a class="header-anchor" href="#聚类分析与主题提取"><span>聚类分析与主题提取</span></a></h3>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># 使用最佳K值进行聚类</span></span>
<span class="line">kmeans <span class="token operator">=</span> KMeans<span class="token punctuation">(</span>n_clusters<span class="token operator">=</span>best_k<span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">df<span class="token punctuation">[</span><span class="token string">'cluster'</span><span class="token punctuation">]</span> <span class="token operator">=</span> kmeans<span class="token punctuation">.</span>fit_predict<span class="token punctuation">(</span>X<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 可视化聚类结果</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">8</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">scatter <span class="token operator">=</span> plt<span class="token punctuation">.</span>scatter<span class="token punctuation">(</span>X_svd<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> X_svd<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">,</span> c<span class="token operator">=</span>df<span class="token punctuation">[</span><span class="token string">'cluster'</span><span class="token punctuation">]</span><span class="token punctuation">,</span> cmap<span class="token operator">=</span><span class="token string">'viridis'</span><span class="token punctuation">,</span> s<span class="token operator">=</span><span class="token number">50</span><span class="token punctuation">,</span> alpha<span class="token operator">=</span><span class="token number">0.8</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'新闻文章聚类结果'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'成分1'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'成分2'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>colorbar<span class="token punctuation">(</span>scatter<span class="token punctuation">,</span> label<span class="token operator">=</span><span class="token string">'簇标签'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>grid<span class="token punctuation">(</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 提取每个簇的关键词</span></span>
<span class="line">feature_names <span class="token operator">=</span> vectorizer<span class="token punctuation">.</span>get_feature_names_out<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">centroids <span class="token operator">=</span> kmeans<span class="token punctuation">.</span>cluster_centers_</span>
<span class="line"></span>
<span class="line"><span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>best_k<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    <span class="token comment"># 获取簇中心的前10个关键词</span></span>
<span class="line">    top_indices <span class="token operator">=</span> centroids<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">.</span>argsort<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token operator">-</span><span class="token number">10</span><span class="token punctuation">:</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">:</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span></span>
<span class="line">    top_keywords <span class="token operator">=</span> <span class="token punctuation">[</span>feature_names<span class="token punctuation">[</span>idx<span class="token punctuation">]</span> <span class="token keyword">for</span> idx <span class="token keyword">in</span> top_indices<span class="token punctuation">]</span></span>
<span class="line">    </span>
<span class="line">    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"簇 </span><span class="token interpolation"><span class="token punctuation">{</span>i<span class="token punctuation">}</span></span><span class="token string"> 的关键词: </span><span class="token interpolation"><span class="token punctuation">{</span><span class="token string">', '</span><span class="token punctuation">.</span>join<span class="token punctuation">(</span>top_keywords<span class="token punctuation">)</span><span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># 显示该簇的示例文章标题</span></span>
<span class="line">    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"簇 </span><span class="token interpolation"><span class="token punctuation">{</span>i<span class="token punctuation">}</span></span><span class="token string"> 的示例文章:"</span></span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token keyword">for</span> title <span class="token keyword">in</span> df<span class="token punctuation">[</span>df<span class="token punctuation">[</span><span class="token string">'cluster'</span><span class="token punctuation">]</span> <span class="token operator">==</span> i<span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">'title'</span><span class="token punctuation">]</span><span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">        <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"- </span><span class="token interpolation"><span class="token punctuation">{</span>title<span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h3 id="业务应用" tabindex="-1"><a class="header-anchor" href="#业务应用"><span>业务应用</span></a></h3>
<p>基于文档聚类结果，可以实现以下应用：</p>
<ol>
<li><strong>自动内容分类</strong>：将新文章自动分配到相应的类别</li>
<li><strong>相关文章推荐</strong>：为用户推荐与当前阅读文章同类的其他文章</li>
<li><strong>主题发现</strong>：识别热门话题和新兴趋势</li>
<li><strong>内容组织</strong>：优化网站导航和内容结构</li>
</ol>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见误区
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>忽略业务背景</strong>：聚类结果需要结合业务知识解释才有意义</li>
      <li><strong>过度依赖自动化</strong>：聚类是辅助工具，不应完全替代人工判断</li>
      <li><strong>忽略数据质量</strong>：垃圾进，垃圾出，数据质量对聚类结果至关重要</li>
      <li><strong>忽略模型更新</strong>：客户行为和市场环境会变化，聚类模型需要定期更新</li>
    </ul>
  </div>
</div>
<h2 id="小结与思考" tabindex="-1"><a class="header-anchor" href="#小结与思考"><span>小结与思考</span></a></h2>
<p>聚类分析在客户分群、异常检测和文档组织等多个领域有广泛应用。通过将数据划分为有意义的群体，企业可以获得宝贵的业务洞察。</p>
<h3 id="关键要点回顾" tabindex="-1"><a class="header-anchor" href="#关键要点回顾"><span>关键要点回顾</span></a></h3>
<ul>
<li>聚类分析可以帮助企业发现数据中的自然分组</li>
<li>从业务问题到聚类方案需要合理的特征选择和预处理</li>
<li>聚类结果的解释需要结合领域知识</li>
<li>聚类分析可以为个性化营销、风险管理等提供支持</li>
</ul>
<h3 id="思考问题" tabindex="-1"><a class="header-anchor" href="#思考问题"><span>思考问题</span></a></h3>
<ol>
<li>如何将聚类结果转化为可操作的业务策略？</li>
<li>在实际应用中，如何评估聚类方案的业务价值？</li>
<li>聚类分析如何与其他数据挖掘技术结合使用？</li>
</ol>
<BackToPath /><div class="practice-link">
  <a href="/projects/clustering.html" class="button">前往实践项目</a>
</div> </div></template>


