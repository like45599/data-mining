<template><div><h1 id="回归模型选择" tabindex="-1"><a class="header-anchor" href="#回归模型选择"><span>回归模型选择</span></a></h1>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>理解不同回归模型的适用场景和局限性</li>
      <li>掌握模型选择的方法和策略</li>
      <li>学习如何使用交叉验证和网格搜索优化模型</li>
      <li>了解模型集成和堆叠技术提高预测性能</li>
    </ul>
  </div>
</div>
<h2 id="回归模型比较" tabindex="-1"><a class="header-anchor" href="#回归模型比较"><span>回归模型比较</span></a></h2>
<p>不同的回归模型有各自的优缺点和适用场景。以下是常见回归模型的比较：</p>
<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>模型</th>
        <th>优点</th>
        <th>缺点</th>
        <th>适用场景</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>线性回归</td>
        <td>
          - 简单易解释<br>
          - 训练速度快<br>
          - 低方差
        </td>
        <td>
          - 假设线性关系<br>
          - 对异常值敏感<br>
          - 无法捕捉非线性关系
        </td>
        <td>
          - 特征与目标变量呈线性关系<br>
          - 需要可解释性<br>
          - 数据量较小
        </td>
      </tr>
      <tr>
        <td>决策树回归</td>
        <td>
          - 可捕捉非线性关系<br>
          - 无需特征缩放<br>
          - 可处理分类特征
        </td>
        <td>
          - 容易过拟合<br>
          - 不稳定<br>
          - 预测不连续
        </td>
        <td>
          - 特征与目标变量呈非线性关系<br>
          - 存在特征交互<br>
          - 需要可解释性
        </td>
      </tr>
      <tr>
        <td>随机森林回归</td>
        <td>
          - 减少过拟合<br>
          - 提供特征重要性<br>
          - 鲁棒性强
        </td>
        <td>
          - 计算复杂度高<br>
          - 可解释性降低<br>
          - 超参数多
        </td>
        <td>
          - 特征与目标变量关系复杂<br>
          - 数据噪声较大<br>
          - 需要稳定性
        </td>
      </tr>
      <tr>
        <td>梯度提升回归</td>
        <td>
          - 高预测精度<br>
          - 可处理不同类型特征<br>
          - 提供特征重要性
        </td>
        <td>
          - 计算复杂度高<br>
          - 容易过拟合<br>
          - 参数调优复杂
        </td>
        <td>
          - 需要高预测精度<br>
          - 有足够计算资源<br>
          - 数据结构复杂
        </td>
      </tr>
      <tr>
        <td>支持向量回归</td>
        <td>
          - 处理非线性关系<br>
          - 对异常值鲁棒<br>
          - 泛化能力强
        </td>
        <td>
          - 计算复杂度高<br>
          - 参数敏感<br>
          - 不适合大数据集
        </td>
        <td>
          - 中小型数据集<br>
          - 存在异常值<br>
          - 需要高泛化能力
        </td>
      </tr>
      <tr>
        <td>神经网络回归</td>
        <td>
          - 处理复杂非线性关系<br>
          - 自动特征提取<br>
          - 适应大数据集
        </td>
        <td>
          - 计算资源需求高<br>
          - 可解释性差<br>
          - 需要大量数据
        </td>
        <td>
          - 复杂非线性关系<br>
          - 大型数据集<br>
          - 不需要高可解释性
        </td>
      </tr>
    </tbody>
  </table>
</div>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>没有一种回归模型在所有场景下都表现最佳，这就是所谓的"没有免费的午餐"定理。模型选择应该基于数据特性、问题需求和计算资源等因素综合考虑。</p>
  </div>
</div>
<h2 id="模型选择策略" tabindex="-1"><a class="header-anchor" href="#模型选择策略"><span>模型选择策略</span></a></h2>
<p>选择合适的回归模型通常遵循以下步骤：</p>
<ol>
<li><strong>理解问题</strong>：明确预测目标和评估指标</li>
<li><strong>探索数据</strong>：分析特征与目标变量的关系</li>
<li><strong>特征工程</strong>：创建、转换和选择特征</li>
<li><strong>基准模型</strong>：建立简单模型作为基准</li>
<li><strong>模型比较</strong>：尝试不同类型的模型</li>
<li><strong>超参数调优</strong>：优化选定模型的参数</li>
<li><strong>模型验证</strong>：使用交叉验证评估模型性能</li>
<li><strong>模型集成</strong>：组合多个模型提高性能</li>
</ol>
<div class="visualization-container">
  <div class="visualization-title">回归模型选择流程</div>
  <div class="visualization-content">
    <img src="/images/regression_model_selection.png" alt="回归模型选择流程">
  </div>
  <div class="visualization-caption">
    图1: 回归模型选择流程。从数据分析开始，经过模型评估和优化，最终选择最佳模型。
  </div>
</div>
<h2 id="使用交叉验证选择模型" tabindex="-1"><a class="header-anchor" href="#使用交叉验证选择模型"><span>使用交叉验证选择模型</span></a></h2>
<p>交叉验证是比较不同模型性能的可靠方法。</p>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">import</span> numpy <span class="token keyword">as</span> np</span>
<span class="line"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd</span>
<span class="line"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>model_selection <span class="token keyword">import</span> cross_val_score<span class="token punctuation">,</span> KFold</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>linear_model <span class="token keyword">import</span> LinearRegression<span class="token punctuation">,</span> Ridge<span class="token punctuation">,</span> Lasso</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>tree <span class="token keyword">import</span> DecisionTreeRegressor</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>ensemble <span class="token keyword">import</span> RandomForestRegressor<span class="token punctuation">,</span> GradientBoostingRegressor</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>svm <span class="token keyword">import</span> SVR</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>neural_network <span class="token keyword">import</span> MLPRegressor</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>preprocessing <span class="token keyword">import</span> StandardScaler</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>pipeline <span class="token keyword">import</span> Pipeline</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 加载数据</span></span>
<span class="line"><span class="token comment"># X, y = load_your_data()</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 创建模型</span></span>
<span class="line">models <span class="token operator">=</span> <span class="token punctuation">{</span></span>
<span class="line">    <span class="token string">'Linear Regression'</span><span class="token punctuation">:</span> LinearRegression<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Ridge'</span><span class="token punctuation">:</span> Ridge<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Lasso'</span><span class="token punctuation">:</span> Lasso<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Decision Tree'</span><span class="token punctuation">:</span> DecisionTreeRegressor<span class="token punctuation">(</span>random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Random Forest'</span><span class="token punctuation">:</span> RandomForestRegressor<span class="token punctuation">(</span>random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Gradient Boosting'</span><span class="token punctuation">:</span> GradientBoostingRegressor<span class="token punctuation">(</span>random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'SVR'</span><span class="token punctuation">:</span> SVR<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Neural Network'</span><span class="token punctuation">:</span> MLPRegressor<span class="token punctuation">(</span>random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token punctuation">}</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 创建管道</span></span>
<span class="line">pipelines <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token punctuation">}</span></span>
<span class="line"><span class="token keyword">for</span> name<span class="token punctuation">,</span> model <span class="token keyword">in</span> models<span class="token punctuation">.</span>items<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    <span class="token keyword">if</span> name <span class="token keyword">in</span> <span class="token punctuation">[</span><span class="token string">'SVR'</span><span class="token punctuation">,</span> <span class="token string">'Ridge'</span><span class="token punctuation">,</span> <span class="token string">'Lasso'</span><span class="token punctuation">,</span> <span class="token string">'Neural Network'</span><span class="token punctuation">]</span><span class="token punctuation">:</span></span>
<span class="line">        pipelines<span class="token punctuation">[</span>name<span class="token punctuation">]</span> <span class="token operator">=</span> Pipeline<span class="token punctuation">(</span><span class="token punctuation">[</span></span>
<span class="line">            <span class="token punctuation">(</span><span class="token string">'scaler'</span><span class="token punctuation">,</span> StandardScaler<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">            <span class="token punctuation">(</span><span class="token string">'model'</span><span class="token punctuation">,</span> model<span class="token punctuation">)</span></span>
<span class="line">        <span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token keyword">else</span><span class="token punctuation">:</span></span>
<span class="line">        pipelines<span class="token punctuation">[</span>name<span class="token punctuation">]</span> <span class="token operator">=</span> Pipeline<span class="token punctuation">(</span><span class="token punctuation">[</span></span>
<span class="line">            <span class="token punctuation">(</span><span class="token string">'model'</span><span class="token punctuation">,</span> model<span class="token punctuation">)</span></span>
<span class="line">        <span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 设置交叉验证</span></span>
<span class="line">kf <span class="token operator">=</span> KFold<span class="token punctuation">(</span>n_splits<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">,</span> shuffle<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 评估模型</span></span>
<span class="line">results <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token punctuation">}</span></span>
<span class="line"><span class="token keyword">for</span> name<span class="token punctuation">,</span> pipeline <span class="token keyword">in</span> pipelines<span class="token punctuation">.</span>items<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    <span class="token comment"># 计算负MSE（sklearn使用负分数表示需要最小化的指标）</span></span>
<span class="line">    mse_scores <span class="token operator">=</span> <span class="token operator">-</span>cross_val_score<span class="token punctuation">(</span>pipeline<span class="token punctuation">,</span> X<span class="token punctuation">,</span> y<span class="token punctuation">,</span> cv<span class="token operator">=</span>kf<span class="token punctuation">,</span> scoring<span class="token operator">=</span><span class="token string">'neg_mean_squared_error'</span><span class="token punctuation">)</span></span>
<span class="line">    r2_scores <span class="token operator">=</span> cross_val_score<span class="token punctuation">(</span>pipeline<span class="token punctuation">,</span> X<span class="token punctuation">,</span> y<span class="token punctuation">,</span> cv<span class="token operator">=</span>kf<span class="token punctuation">,</span> scoring<span class="token operator">=</span><span class="token string">'r2'</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    results<span class="token punctuation">[</span>name<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token punctuation">{</span></span>
<span class="line">        <span class="token string">'MSE'</span><span class="token punctuation">:</span> <span class="token punctuation">{</span></span>
<span class="line">            <span class="token string">'mean'</span><span class="token punctuation">:</span> mse_scores<span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">            <span class="token string">'std'</span><span class="token punctuation">:</span> mse_scores<span class="token punctuation">.</span>std<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">        <span class="token punctuation">}</span><span class="token punctuation">,</span></span>
<span class="line">        <span class="token string">'R²'</span><span class="token punctuation">:</span> <span class="token punctuation">{</span></span>
<span class="line">            <span class="token string">'mean'</span><span class="token punctuation">:</span> r2_scores<span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">            <span class="token string">'std'</span><span class="token punctuation">:</span> r2_scores<span class="token punctuation">.</span>std<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">        <span class="token punctuation">}</span></span>
<span class="line">    <span class="token punctuation">}</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 显示结果</span></span>
<span class="line"><span class="token keyword">for</span> name<span class="token punctuation">,</span> metrics <span class="token keyword">in</span> results<span class="token punctuation">.</span>items<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"</span><span class="token interpolation"><span class="token punctuation">{</span>name<span class="token punctuation">}</span></span><span class="token string">:"</span></span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"  MSE: </span><span class="token interpolation"><span class="token punctuation">{</span>metrics<span class="token punctuation">[</span><span class="token string">'MSE'</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">'mean'</span><span class="token punctuation">]</span><span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string"> ± </span><span class="token interpolation"><span class="token punctuation">{</span>metrics<span class="token punctuation">[</span><span class="token string">'MSE'</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">'std'</span><span class="token punctuation">]</span><span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"  R²: </span><span class="token interpolation"><span class="token punctuation">{</span>metrics<span class="token punctuation">[</span><span class="token string">'R²'</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">'mean'</span><span class="token punctuation">]</span><span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string"> ± </span><span class="token interpolation"><span class="token punctuation">{</span>metrics<span class="token punctuation">[</span><span class="token string">'R²'</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">'std'</span><span class="token punctuation">]</span><span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 可视化比较</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">names <span class="token operator">=</span> <span class="token builtin">list</span><span class="token punctuation">(</span>results<span class="token punctuation">.</span>keys<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">mse_means <span class="token operator">=</span> <span class="token punctuation">[</span>results<span class="token punctuation">[</span>name<span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">'MSE'</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">'mean'</span><span class="token punctuation">]</span> <span class="token keyword">for</span> name <span class="token keyword">in</span> names<span class="token punctuation">]</span></span>
<span class="line">mse_stds <span class="token operator">=</span> <span class="token punctuation">[</span>results<span class="token punctuation">[</span>name<span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">'MSE'</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">'std'</span><span class="token punctuation">]</span> <span class="token keyword">for</span> name <span class="token keyword">in</span> names<span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line">plt<span class="token punctuation">.</span>barh<span class="token punctuation">(</span>names<span class="token punctuation">,</span> mse_means<span class="token punctuation">,</span> xerr<span class="token operator">=</span>mse_stds<span class="token punctuation">,</span> alpha<span class="token operator">=</span><span class="token number">0.7</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'MSE'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'不同回归模型的MSE比较'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>grid<span class="token punctuation">(</span>axis<span class="token operator">=</span><span class="token string">'x'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>tight_layout<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h2 id="超参数优化" tabindex="-1"><a class="header-anchor" href="#超参数优化"><span>超参数优化</span></a></h2>
<p>使用网格搜索或随机搜索优化模型超参数。</p>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>model_selection <span class="token keyword">import</span> GridSearchCV</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 假设我们选择了随机森林作为最佳模型</span></span>
<span class="line">best_model <span class="token operator">=</span> RandomForestRegressor<span class="token punctuation">(</span>random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 定义参数网格</span></span>
<span class="line">param_grid <span class="token operator">=</span> <span class="token punctuation">{</span></span>
<span class="line">    <span class="token string">'n_estimators'</span><span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token number">50</span><span class="token punctuation">,</span> <span class="token number">100</span><span class="token punctuation">,</span> <span class="token number">200</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'max_depth'</span><span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token boolean">None</span><span class="token punctuation">,</span> <span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">20</span><span class="token punctuation">,</span> <span class="token number">30</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'min_samples_split'</span><span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token number">2</span><span class="token punctuation">,</span> <span class="token number">5</span><span class="token punctuation">,</span> <span class="token number">10</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'min_samples_leaf'</span><span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">,</span> <span class="token number">4</span><span class="token punctuation">]</span></span>
<span class="line"><span class="token punctuation">}</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 创建网格搜索</span></span>
<span class="line">grid_search <span class="token operator">=</span> GridSearchCV<span class="token punctuation">(</span></span>
<span class="line">    best_model<span class="token punctuation">,</span></span>
<span class="line">    param_grid<span class="token punctuation">,</span></span>
<span class="line">    cv<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">,</span></span>
<span class="line">    scoring<span class="token operator">=</span><span class="token string">'neg_mean_squared_error'</span><span class="token punctuation">,</span></span>
<span class="line">    n_jobs<span class="token operator">=</span><span class="token operator">-</span><span class="token number">1</span></span>
<span class="line"><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 训练</span></span>
<span class="line">grid_search<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X<span class="token punctuation">,</span> y<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 最佳参数</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"最佳参数:"</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>grid_search<span class="token punctuation">.</span>best_params_<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 最佳性能</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"最佳MSE: </span><span class="token interpolation"><span class="token punctuation">{</span><span class="token operator">-</span>grid_search<span class="token punctuation">.</span>best_score_<span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 使用最佳参数创建模型</span></span>
<span class="line">best_rf <span class="token operator">=</span> RandomForestRegressor<span class="token punctuation">(</span></span>
<span class="line">    n_estimators<span class="token operator">=</span>grid_search<span class="token punctuation">.</span>best_params_<span class="token punctuation">[</span><span class="token string">'n_estimators'</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">    max_depth<span class="token operator">=</span>grid_search<span class="token punctuation">.</span>best_params_<span class="token punctuation">[</span><span class="token string">'max_depth'</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">    min_samples_split<span class="token operator">=</span>grid_search<span class="token punctuation">.</span>best_params_<span class="token punctuation">[</span><span class="token string">'min_samples_split'</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">    min_samples_leaf<span class="token operator">=</span>grid_search<span class="token punctuation">.</span>best_params_<span class="token punctuation">[</span><span class="token string">'min_samples_leaf'</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">    random_state<span class="token operator">=</span><span class="token number">42</span></span>
<span class="line"><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h2 id="模型集成" tabindex="-1"><a class="header-anchor" href="#模型集成"><span>模型集成</span></a></h2>
<p>组合多个模型可以提高预测性能。</p>
<h3 id="投票回归" tabindex="-1"><a class="header-anchor" href="#投票回归"><span>投票回归</span></a></h3>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>ensemble <span class="token keyword">import</span> VotingRegressor</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 选择表现最好的几个模型</span></span>
<span class="line">estimators <span class="token operator">=</span> <span class="token punctuation">[</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'rf'</span><span class="token punctuation">,</span> RandomForestRegressor<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'gbr'</span><span class="token punctuation">,</span> GradientBoostingRegressor<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'svr'</span><span class="token punctuation">,</span> SVR<span class="token punctuation">(</span>kernel<span class="token operator">=</span><span class="token string">'rbf'</span><span class="token punctuation">,</span> C<span class="token operator">=</span><span class="token number">10</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 创建投票回归器</span></span>
<span class="line">voting_reg <span class="token operator">=</span> VotingRegressor<span class="token punctuation">(</span>estimators<span class="token operator">=</span>estimators<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 评估</span></span>
<span class="line">scores <span class="token operator">=</span> cross_val_score<span class="token punctuation">(</span>voting_reg<span class="token punctuation">,</span> X<span class="token punctuation">,</span> y<span class="token punctuation">,</span> cv<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">,</span> scoring<span class="token operator">=</span><span class="token string">'neg_mean_squared_error'</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"投票回归MSE: </span><span class="token interpolation"><span class="token punctuation">{</span><span class="token operator">-</span>scores<span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string"> ± </span><span class="token interpolation"><span class="token punctuation">{</span>scores<span class="token punctuation">.</span>std<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h3 id="堆叠回归" tabindex="-1"><a class="header-anchor" href="#堆叠回归"><span>堆叠回归</span></a></h3>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>ensemble <span class="token keyword">import</span> StackingRegressor</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 定义基础模型</span></span>
<span class="line">base_models <span class="token operator">=</span> <span class="token punctuation">[</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'rf'</span><span class="token punctuation">,</span> RandomForestRegressor<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'gbr'</span><span class="token punctuation">,</span> GradientBoostingRegressor<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token punctuation">(</span><span class="token string">'svr'</span><span class="token punctuation">,</span> Pipeline<span class="token punctuation">(</span><span class="token punctuation">[</span></span>
<span class="line">        <span class="token punctuation">(</span><span class="token string">'scaler'</span><span class="token punctuation">,</span> StandardScaler<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">        <span class="token punctuation">(</span><span class="token string">'svr'</span><span class="token punctuation">,</span> SVR<span class="token punctuation">(</span>kernel<span class="token operator">=</span><span class="token string">'rbf'</span><span class="token punctuation">,</span> C<span class="token operator">=</span><span class="token number">10</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 定义元模型</span></span>
<span class="line">meta_model <span class="token operator">=</span> Ridge<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 创建堆叠回归器</span></span>
<span class="line">stacking_reg <span class="token operator">=</span> StackingRegressor<span class="token punctuation">(</span></span>
<span class="line">    estimators<span class="token operator">=</span>base_models<span class="token punctuation">,</span></span>
<span class="line">    final_estimator<span class="token operator">=</span>meta_model<span class="token punctuation">,</span></span>
<span class="line">    cv<span class="token operator">=</span><span class="token number">5</span></span>
<span class="line"><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 评估</span></span>
<span class="line">scores <span class="token operator">=</span> cross_val_score<span class="token punctuation">(</span>stacking_reg<span class="token punctuation">,</span> X<span class="token punctuation">,</span> y<span class="token punctuation">,</span> cv<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">,</span> scoring<span class="token operator">=</span><span class="token string">'neg_mean_squared_error'</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"堆叠回归MSE: </span><span class="token interpolation"><span class="token punctuation">{</span><span class="token operator">-</span>scores<span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string"> ± </span><span class="token interpolation"><span class="token punctuation">{</span>scores<span class="token punctuation">.</span>std<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h2 id="特征重要性分析" tabindex="-1"><a class="header-anchor" href="#特征重要性分析"><span>特征重要性分析</span></a></h2>
<p>了解特征对模型预测的影响可以帮助我们选择更合适的模型。</p>
<div class="code-example">
  <div class="code-example__title">代码示例</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># 训练随机森林模型</span></span>
<span class="line">rf <span class="token operator">=</span> RandomForestRegressor<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">rf<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X<span class="token punctuation">,</span> y<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 获取特征重要性</span></span>
<span class="line">importances <span class="token operator">=</span> rf<span class="token punctuation">.</span>feature_importances_</span>
<span class="line">indices <span class="token operator">=</span> np<span class="token punctuation">.</span>argsort<span class="token punctuation">(</span>importances<span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">:</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 可视化特征重要性</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'特征重要性'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>bar<span class="token punctuation">(</span><span class="token builtin">range</span><span class="token punctuation">(</span>X<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span> importances<span class="token punctuation">[</span>indices<span class="token punctuation">]</span><span class="token punctuation">,</span> align<span class="token operator">=</span><span class="token string">'center'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xticks<span class="token punctuation">(</span><span class="token builtin">range</span><span class="token punctuation">(</span>X<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token punctuation">[</span><span class="token string-interpolation"><span class="token string">f'特征 </span><span class="token interpolation"><span class="token punctuation">{</span>i<span class="token punctuation">}</span></span><span class="token string">'</span></span> <span class="token keyword">for</span> i <span class="token keyword">in</span> indices<span class="token punctuation">]</span><span class="token punctuation">,</span> rotation<span class="token operator">=</span><span class="token number">90</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>tight_layout<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h2 id="模型解释性" tabindex="-1"><a class="header-anchor" href="#模型解释性"><span>模型解释性</span></a></h2>
<p>在某些应用中，模型的可解释性与预测性能同样重要。</p>
<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>模型</th>
        <th>解释性</th>
        <th>解释方法</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>线性回归</td>
        <td>高</td>
        <td>系数大小和符号</td>
      </tr>
      <tr>
        <td>决策树</td>
        <td>中高</td>
        <td>树结构和决策路径</td>
      </tr>
      <tr>
        <td>随机森林</td>
        <td>中</td>
        <td>特征重要性</td>
      </tr>
      <tr>
        <td>梯度提升树</td>
        <td>中</td>
        <td>特征重要性、部分依赖图</td>
      </tr>
      <tr>
        <td>支持向量回归</td>
        <td>低</td>
        <td>SHAP值、排列重要性</td>
      </tr>
      <tr>
        <td>神经网络</td>
        <td>很低</td>
        <td>SHAP值、排列重要性</td>
      </tr>
    </tbody>
  </table>
</div>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见误区
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>过度追求复杂模型</strong>：简单模型可能更稳定且易于维护</li>
      <li><strong>忽略领域知识</strong>：领域知识可以指导特征工程和模型选择</li>
      <li><strong>过度依赖单一评估指标</strong>：应考虑多种指标和业务需求</li>
      <li><strong>忽略计算成本</strong>：在生产环境中，模型的训练和预测时间也很重要</li>
    </ul>
  </div>
</div>
<h2 id="小结与思考" tabindex="-1"><a class="header-anchor" href="#小结与思考"><span>小结与思考</span></a></h2>
<p>选择合适的回归模型是数据科学工作流程中的关键步骤，需要综合考虑多种因素。</p>
<h3 id="关键要点回顾" tabindex="-1"><a class="header-anchor" href="#关键要点回顾"><span>关键要点回顾</span></a></h3>
<ul>
<li>不同回归模型有各自的优缺点和适用场景</li>
<li>交叉验证是比较模型性能的可靠方法</li>
<li>超参数调优可以显著提高模型性能</li>
<li>模型集成通常比单个模型表现更好</li>
<li>模型选择应平衡预测性能、解释性和计算成本</li>
</ul>
<h3 id="思考问题" tabindex="-1"><a class="header-anchor" href="#思考问题"><span>思考问题</span></a></h3>
<ol>
<li>在什么情况下应该选择简单模型而非复杂模型？</li>
<li>如何平衡模型的预测性能和解释性？</li>
<li>模型集成为什么通常能提高预测性能？</li>
</ol>
<BackToPath /><div class="practice-link">
  <a href="/projects/regression.html" class="button">前往实践项目</a>
</div> </div></template>


