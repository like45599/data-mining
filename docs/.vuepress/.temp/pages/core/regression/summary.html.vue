<template><div><h1 id="回归分析总结" tabindex="-1"><a class="header-anchor" href="#回归分析总结"><span>回归分析总结</span></a></h1>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>本节要点
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>回顾回归分析的核心概念和方法</li>
      <li>总结不同回归模型的优缺点和适用场景</li>
      <li>了解回归分析的最佳实践和常见陷阱</li>
      <li>探讨回归分析的未来发展趋势</li>
    </ul>
  </div>
</div>
<h2 id="回归分析核心概念回顾" tabindex="-1"><a class="header-anchor" href="#回归分析核心概念回顾"><span>回归分析核心概念回顾</span></a></h2>
<p>回归分析是预测连续目标变量的重要方法，其核心概念包括：</p>
<h3 id="基本原理" tabindex="-1"><a class="header-anchor" href="#基本原理"><span>基本原理</span></a></h3>
<p>回归分析的目标是建立自变量（特征）和因变量（目标）之间的关系模型，用于预测和解释。</p>
<h3 id="关键假设" tabindex="-1"><a class="header-anchor" href="#关键假设"><span>关键假设</span></a></h3>
<p>不同回归模型有不同的假设，但常见的假设包括：</p>
<ul>
<li><strong>线性关系</strong>：线性回归假设特征和目标之间存在线性关系</li>
<li><strong>独立性</strong>：观测值之间相互独立</li>
<li><strong>同方差性</strong>：误差项具有恒定方差</li>
<li><strong>正态性</strong>：误差项服从正态分布（对某些模型而言）</li>
</ul>
<h3 id="模型评估" tabindex="-1"><a class="header-anchor" href="#模型评估"><span>模型评估</span></a></h3>
<p>评估回归模型性能的常用指标包括：</p>
<ul>
<li><strong>均方误差(MSE)</strong>：预测值与实际值差异的平方的平均值</li>
<li><strong>均方根误差(RMSE)</strong>：MSE的平方根，与目标变量单位相同</li>
<li><strong>平均绝对误差(MAE)</strong>：预测值与实际值差异的绝对值的平均值</li>
<li><strong>决定系数(R²)</strong>：模型解释的目标变量方差比例</li>
</ul>
<div class="visualization-container">
  <div class="visualization-title">回归分析流程</div>
  <div class="visualization-content">
    <img src="/images/regression_workflow.png" alt="回归分析流程">
  </div>
  <div class="visualization-caption">
    图1: 回归分析的完整流程，从问题定义到模型部署。
  </div>
</div>
<h2 id="回归模型比较" tabindex="-1"><a class="header-anchor" href="#回归模型比较"><span>回归模型比较</span></a></h2>
<p>以下是主要回归模型的比较总结：</p>
<div class="table-container">
  <table>
    <thead>
      <tr>
        <th>模型</th>
        <th>优点</th>
        <th>缺点</th>
        <th>适用场景</th>
        <th>复杂度</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>线性回归</td>
        <td>简单易解释，训练快</td>
        <td>只能捕捉线性关系，对异常值敏感</td>
        <td>线性关系，需要高解释性</td>
        <td>低</td>
      </tr>
      <tr>
        <td>岭回归</td>
        <td>处理多重共线性，减少过拟合</td>
        <td>需要调优正则化参数</td>
        <td>特征间高度相关</td>
        <td>低</td>
      </tr>
      <tr>
        <td>Lasso回归</td>
        <td>特征选择，稀疏解</td>
        <td>可能过度惩罚重要特征</td>
        <td>需要特征选择</td>
        <td>低</td>
      </tr>
      <tr>
        <td>决策树回归</td>
        <td>捕捉非线性关系，易于理解</td>
        <td>容易过拟合，不稳定</td>
        <td>非线性关系，需要可解释性</td>
        <td>中</td>
      </tr>
      <tr>
        <td>随机森林回归</td>
        <td>减少过拟合，提供特征重要性</td>
        <td>计算复杂度高，可解释性降低</td>
        <td>复杂关系，需要稳定性</td>
        <td>中高</td>
      </tr>
      <tr>
        <td>梯度提升回归</td>
        <td>高预测精度，处理不同类型特征</td>
        <td>参数调优复杂，计算成本高</td>
        <td>需要高预测精度</td>
        <td>高</td>
      </tr>
      <tr>
        <td>支持向量回归</td>
        <td>处理非线性关系，对异常值鲁棒</td>
        <td>参数敏感，不适合大数据集</td>
        <td>中小型数据集，存在异常值</td>
        <td>中高</td>
      </tr>
      <tr>
        <td>神经网络回归</td>
        <td>处理复杂非线性关系，自动特征提取</td>
        <td>需要大量数据，可解释性低</td>
        <td>大数据集，复杂关系</td>
        <td>很高</td>
      </tr>
    </tbody>
  </table>
</div>
<h2 id="回归分析最佳实践" tabindex="-1"><a class="header-anchor" href="#回归分析最佳实践"><span>回归分析最佳实践</span></a></h2>
<p>以下是进行回归分析的一些最佳实践：</p>
<h3 id="数据预处理" tabindex="-1"><a class="header-anchor" href="#数据预处理"><span>数据预处理</span></a></h3>
<ol>
<li><strong>处理缺失值</strong>：使用均值、中位数填充或高级插补方法</li>
<li><strong>处理异常值</strong>：识别并处理异常值，如截断或变换</li>
<li><strong>特征缩放</strong>：标准化或归一化特征，特别是对距离敏感的模型</li>
<li><strong>特征编码</strong>：将分类特征转换为数值表示</li>
</ol>
<h3 id="特征工程" tabindex="-1"><a class="header-anchor" href="#特征工程"><span>特征工程</span></a></h3>
<ol>
<li><strong>特征选择</strong>：移除不相关或冗余特征，如使用Lasso或特征重要性</li>
<li><strong>特征变换</strong>：应用对数、多项式等变换捕捉非线性关系</li>
<li><strong>特征交互</strong>：创建特征交互项捕捉特征间的相互作用</li>
<li><strong>降维</strong>：使用PCA等方法减少特征数量</li>
</ol>
<h3 id="模型选择与调优" tabindex="-1"><a class="header-anchor" href="#模型选择与调优"><span>模型选择与调优</span></a></h3>
<ol>
<li><strong>从简单模型开始</strong>：先尝试线性模型作为基准</li>
<li><strong>交叉验证</strong>：使用k折交叉验证评估模型性能</li>
<li><strong>超参数调优</strong>：使用网格搜索或随机搜索优化超参数</li>
<li><strong>模型集成</strong>：组合多个模型提高预测性能</li>
</ol>
<h3 id="模型评估与解释" tabindex="-1"><a class="header-anchor" href="#模型评估与解释"><span>模型评估与解释</span></a></h3>
<ol>
<li><strong>多指标评估</strong>：使用多种评估指标全面评估模型</li>
<li><strong>残差分析</strong>：检查残差分布识别模型问题</li>
<li><strong>特征重要性</strong>：分析特征对预测的贡献</li>
<li><strong>可解释性工具</strong>：使用SHAP值等工具解释复杂模型</li>
</ol>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">⚠️</span>常见陷阱
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>数据泄露</strong>：测试数据信息不当地用于训练</li>
      <li><strong>过度拟合</strong>：模型过于复杂，无法泛化到新数据</li>
      <li><strong>忽略特征工程</strong>：直接使用原始特征而不进行适当变换</li>
      <li><strong>忽略领域知识</strong>：未将业务理解融入模型构建过程</li>
      <li><strong>过度依赖单一指标</strong>：仅使用一种指标评估模型</li>
    </ul>
  </div>
</div>
<h2 id="回归分析的未来趋势" tabindex="-1"><a class="header-anchor" href="#回归分析的未来趋势"><span>回归分析的未来趋势</span></a></h2>
<p>回归分析领域正在不断发展，以下是一些值得关注的趋势：</p>
<h3 id="自动化机器学习-automl" tabindex="-1"><a class="header-anchor" href="#自动化机器学习-automl"><span>自动化机器学习(AutoML)</span></a></h3>
<p>AutoML工具自动执行特征工程、模型选择和超参数调优，使非专家也能构建高质量回归模型。</p>
<h3 id="可解释人工智能-xai" tabindex="-1"><a class="header-anchor" href="#可解释人工智能-xai"><span>可解释人工智能(XAI)</span></a></h3>
<p>随着模型复杂度增加，可解释性变得越来越重要。新的方法和工具正在开发，以解释复杂回归模型的预测。</p>
<h3 id="联邦学习" tabindex="-1"><a class="header-anchor" href="#联邦学习"><span>联邦学习</span></a></h3>
<p>联邦学习允许在不共享原始数据的情况下训练回归模型，解决数据隐私和安全问题。</p>
<h3 id="神经网络架构搜索-nas" tabindex="-1"><a class="header-anchor" href="#神经网络架构搜索-nas"><span>神经网络架构搜索(NAS)</span></a></h3>
<p>自动设计最适合特定回归问题的神经网络架构，减少人工试错。</p>
<h3 id="因果推断" tabindex="-1"><a class="header-anchor" href="#因果推断"><span>因果推断</span></a></h3>
<p>从相关性到因果关系的转变，使回归分析不仅能预测还能解释因果关系。</p>
<div class="code-example">
  <div class="code-example__title">代码示例：回归分析工作流</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">import</span> numpy <span class="token keyword">as</span> np</span>
<span class="line"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd</span>
<span class="line"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>model_selection <span class="token keyword">import</span> train_test_split<span class="token punctuation">,</span> cross_val_score<span class="token punctuation">,</span> GridSearchCV</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>preprocessing <span class="token keyword">import</span> StandardScaler</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>pipeline <span class="token keyword">import</span> Pipeline</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>linear_model <span class="token keyword">import</span> LinearRegression<span class="token punctuation">,</span> Ridge<span class="token punctuation">,</span> Lasso</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>ensemble <span class="token keyword">import</span> RandomForestRegressor<span class="token punctuation">,</span> GradientBoostingRegressor</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>metrics <span class="token keyword">import</span> mean_squared_error<span class="token punctuation">,</span> r2_score</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 1. 加载数据</span></span>
<span class="line"><span class="token comment"># data = pd.read_csv('your_data.csv')</span></span>
<span class="line"><span class="token comment"># X = data.drop('target', axis=1)</span></span>
<span class="line"><span class="token comment"># y = data['target']</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 2. 数据分割</span></span>
<span class="line">X_train<span class="token punctuation">,</span> X_test<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> y_test <span class="token operator">=</span> train_test_split<span class="token punctuation">(</span>X<span class="token punctuation">,</span> y<span class="token punctuation">,</span> test_size<span class="token operator">=</span><span class="token number">0.2</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 3. 创建模型管道</span></span>
<span class="line">models <span class="token operator">=</span> <span class="token punctuation">{</span></span>
<span class="line">    <span class="token string">'Linear Regression'</span><span class="token punctuation">:</span> Pipeline<span class="token punctuation">(</span><span class="token punctuation">[</span></span>
<span class="line">        <span class="token punctuation">(</span><span class="token string">'scaler'</span><span class="token punctuation">,</span> StandardScaler<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">        <span class="token punctuation">(</span><span class="token string">'model'</span><span class="token punctuation">,</span> LinearRegression<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Ridge'</span><span class="token punctuation">:</span> Pipeline<span class="token punctuation">(</span><span class="token punctuation">[</span></span>
<span class="line">        <span class="token punctuation">(</span><span class="token string">'scaler'</span><span class="token punctuation">,</span> StandardScaler<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">        <span class="token punctuation">(</span><span class="token string">'model'</span><span class="token punctuation">,</span> Ridge<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Random Forest'</span><span class="token punctuation">:</span> Pipeline<span class="token punctuation">(</span><span class="token punctuation">[</span></span>
<span class="line">        <span class="token punctuation">(</span><span class="token string">'model'</span><span class="token punctuation">,</span> RandomForestRegressor<span class="token punctuation">(</span>random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Gradient Boosting'</span><span class="token punctuation">:</span> Pipeline<span class="token punctuation">(</span><span class="token punctuation">[</span></span>
<span class="line">        <span class="token punctuation">(</span><span class="token string">'model'</span><span class="token punctuation">,</span> GradientBoostingRegressor<span class="token punctuation">(</span>random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token punctuation">}</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 4. 交叉验证评估</span></span>
<span class="line">results <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token punctuation">}</span></span>
<span class="line"><span class="token keyword">for</span> name<span class="token punctuation">,</span> model <span class="token keyword">in</span> models<span class="token punctuation">.</span>items<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    scores <span class="token operator">=</span> cross_val_score<span class="token punctuation">(</span>model<span class="token punctuation">,</span> X_train<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> cv<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">,</span> scoring<span class="token operator">=</span><span class="token string">'neg_mean_squared_error'</span><span class="token punctuation">)</span></span>
<span class="line">    rmse_scores <span class="token operator">=</span> np<span class="token punctuation">.</span>sqrt<span class="token punctuation">(</span><span class="token operator">-</span>scores<span class="token punctuation">)</span></span>
<span class="line">    results<span class="token punctuation">[</span>name<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token punctuation">{</span></span>
<span class="line">        <span class="token string">'mean_rmse'</span><span class="token punctuation">:</span> rmse_scores<span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">        <span class="token string">'std_rmse'</span><span class="token punctuation">:</span> rmse_scores<span class="token punctuation">.</span>std<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    <span class="token punctuation">}</span></span>
<span class="line">    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"</span><span class="token interpolation"><span class="token punctuation">{</span>name<span class="token punctuation">}</span></span><span class="token string">: RMSE = </span><span class="token interpolation"><span class="token punctuation">{</span>rmse_scores<span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string"> (±</span><span class="token interpolation"><span class="token punctuation">{</span>rmse_scores<span class="token punctuation">.</span>std<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string">)"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 5. 选择最佳模型（示例）</span></span>
<span class="line">best_model_name <span class="token operator">=</span> <span class="token builtin">min</span><span class="token punctuation">(</span>results<span class="token punctuation">,</span> key<span class="token operator">=</span><span class="token keyword">lambda</span> x<span class="token punctuation">:</span> results<span class="token punctuation">[</span>x<span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">'mean_rmse'</span><span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">best_model <span class="token operator">=</span> models<span class="token punctuation">[</span>best_model_name<span class="token punctuation">]</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"\n最佳模型: </span><span class="token interpolation"><span class="token punctuation">{</span>best_model_name<span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 6. 超参数调优（示例：梯度提升）</span></span>
<span class="line"><span class="token keyword">if</span> best_model_name <span class="token operator">==</span> <span class="token string">'Gradient Boosting'</span><span class="token punctuation">:</span></span>
<span class="line">    param_grid <span class="token operator">=</span> <span class="token punctuation">{</span></span>
<span class="line">        <span class="token string">'model__n_estimators'</span><span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token number">50</span><span class="token punctuation">,</span> <span class="token number">100</span><span class="token punctuation">,</span> <span class="token number">200</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">        <span class="token string">'model__learning_rate'</span><span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token number">0.01</span><span class="token punctuation">,</span> <span class="token number">0.1</span><span class="token punctuation">,</span> <span class="token number">0.2</span><span class="token punctuation">]</span><span class="token punctuation">,</span></span>
<span class="line">        <span class="token string">'model__max_depth'</span><span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">5</span><span class="token punctuation">,</span> <span class="token number">7</span><span class="token punctuation">]</span></span>
<span class="line">    <span class="token punctuation">}</span></span>
<span class="line">    grid_search <span class="token operator">=</span> GridSearchCV<span class="token punctuation">(</span></span>
<span class="line">        models<span class="token punctuation">[</span><span class="token string">'Gradient Boosting'</span><span class="token punctuation">]</span><span class="token punctuation">,</span> param_grid<span class="token punctuation">,</span> cv<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">,</span> </span>
<span class="line">        scoring<span class="token operator">=</span><span class="token string">'neg_mean_squared_error'</span><span class="token punctuation">,</span> n_jobs<span class="token operator">=</span><span class="token operator">-</span><span class="token number">1</span></span>
<span class="line">    <span class="token punctuation">)</span></span>
<span class="line">    grid_search<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train<span class="token punctuation">,</span> y_train<span class="token punctuation">)</span></span>
<span class="line">    best_model <span class="token operator">=</span> grid_search<span class="token punctuation">.</span>best_estimator_</span>
<span class="line">    <span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"最佳参数: </span><span class="token interpolation"><span class="token punctuation">{</span>grid_search<span class="token punctuation">.</span>best_params_<span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 7. 在测试集上评估</span></span>
<span class="line">best_model<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train<span class="token punctuation">,</span> y_train<span class="token punctuation">)</span></span>
<span class="line">y_pred <span class="token operator">=</span> best_model<span class="token punctuation">.</span>predict<span class="token punctuation">(</span>X_test<span class="token punctuation">)</span></span>
<span class="line">mse <span class="token operator">=</span> mean_squared_error<span class="token punctuation">(</span>y_test<span class="token punctuation">,</span> y_pred<span class="token punctuation">)</span></span>
<span class="line">rmse <span class="token operator">=</span> np<span class="token punctuation">.</span>sqrt<span class="token punctuation">(</span>mse<span class="token punctuation">)</span></span>
<span class="line">r2 <span class="token operator">=</span> r2_score<span class="token punctuation">(</span>y_test<span class="token punctuation">,</span> y_pred<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"\n测试集性能:"</span></span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"  RMSE: </span><span class="token interpolation"><span class="token punctuation">{</span>rmse<span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"  R²: </span><span class="token interpolation"><span class="token punctuation">{</span>r2<span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 8. 可视化结果</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>scatter<span class="token punctuation">(</span>y_test<span class="token punctuation">,</span> y_pred<span class="token punctuation">,</span> alpha<span class="token operator">=</span><span class="token number">0.7</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span><span class="token punctuation">[</span>y_test<span class="token punctuation">.</span><span class="token builtin">min</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> y_test<span class="token punctuation">.</span><span class="token builtin">max</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token punctuation">[</span>y_test<span class="token punctuation">.</span><span class="token builtin">min</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> y_test<span class="token punctuation">.</span><span class="token builtin">max</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token string">'k--'</span><span class="token punctuation">,</span> lw<span class="token operator">=</span><span class="token number">2</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'实际值'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'预测值'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'预测值 vs 实际值'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>grid<span class="token punctuation">(</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h2 id="小结与思考" tabindex="-1"><a class="header-anchor" href="#小结与思考"><span>小结与思考</span></a></h2>
<p>回归分析是数据科学中的基础技术，从简单的线性回归到复杂的集成方法和神经网络，为我们提供了丰富的工具来解决预测问题。</p>
<h3 id="关键要点回顾" tabindex="-1"><a class="header-anchor" href="#关键要点回顾"><span>关键要点回顾</span></a></h3>
<ul>
<li>回归分析的目标是建立特征和连续目标变量之间的关系</li>
<li>不同回归模型有各自的优缺点和适用场景</li>
<li>数据预处理和特征工程对模型性能至关重要</li>
<li>模型评估应使用多种指标和交叉验证</li>
<li>模型解释和业务应用是回归分析的最终目标</li>
</ul>
<h3 id="思考问题" tabindex="-1"><a class="header-anchor" href="#思考问题"><span>思考问题</span></a></h3>
<ol>
<li>如何在实际项目中平衡模型复杂度、性能和可解释性？</li>
<li>回归分析如何与其他机器学习技术（如分类、聚类）结合使用？</li>
<li>在大数据和计算资源有限的情况下，如何选择合适的回归方法？</li>
</ol>
<BackToPath /><div class="practice-link">
  <a href="/projects/regression.html" class="button">前往实践项目</a>
</div> </div></template>


