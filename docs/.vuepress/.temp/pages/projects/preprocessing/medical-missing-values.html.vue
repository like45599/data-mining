<template><div><h1 id="医疗数据缺失值处理" tabindex="-1"><a class="header-anchor" href="#医疗数据缺失值处理"><span>医疗数据缺失值处理</span></a></h1>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>项目概述
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>难度</strong>：中级</li>
      <li><strong>类型</strong>：缺失值处理</li>
      <!-- <li><strong>预计时间</strong>：5-7小时</li> -->
      <li><strong>技能点</strong>：缺失值模式分析、多重插补、KNN填充、模型预测填充</li>
      <li><strong>对应知识模块</strong>：<a href="/core/preprocessing/data-presentation.html">数据预处理</a></li>
    </ul>
  </div>
</div>
<h2 id="项目背景" tabindex="-1"><a class="header-anchor" href="#项目背景"><span>项目背景</span></a></h2>
<p>医疗数据在临床研究、疾病预测和治疗方案优化中起着关键作用。然而，医疗数据集通常包含大量缺失值，这可能是由于设备故障、患者未完成所有检查、记录错误或数据输入问题等原因造成的。不当的缺失值处理可能导致偏差的结论，影响医疗决策的准确性。</p>
<p>在这个项目中，我们将处理一个包含多种缺失值的医疗数据集，比较不同缺失值处理方法的效果，并为后续的疾病预测模型准备高质量的数据。</p>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>你知道吗？
  </div>
  <div class="knowledge-card__content">
    <p>医疗数据中的缺失值通常不是随机发生的。例如，某些检查可能只对特定症状的患者进行，导致数据的缺失与患者的健康状况相关。这种非随机缺失模式需要特殊处理，以避免引入偏差。</p>
  </div>
</div>
<h2 id="数据集介绍" tabindex="-1"><a class="header-anchor" href="#数据集介绍"><span>数据集介绍</span></a></h2>
<p>本项目使用的数据集包含5,000名患者的医疗记录，包括以下字段：</p>
<ul>
<li><strong>patient_id</strong>：患者ID</li>
<li><strong>age</strong>：年龄</li>
<li><strong>gender</strong>：性别</li>
<li><strong>bmi</strong>：体重指数</li>
<li><strong>blood_pressure_systolic</strong>：收缩压</li>
<li><strong>blood_pressure_diastolic</strong>：舒张压</li>
<li><strong>heart_rate</strong>：心率</li>
<li><strong>cholesterol</strong>：胆固醇水平</li>
<li><strong>glucose</strong>：血糖水平</li>
<li><strong>smoking</strong>：吸烟状态</li>
<li><strong>alcohol_consumption</strong>：酒精消费水平</li>
<li><strong>physical_activity</strong>：身体活动水平</li>
<li><strong>family_history</strong>：家族病史</li>
<li><strong>medication</strong>：当前用药情况</li>
<li><strong>diagnosis</strong>：诊断结果</li>
</ul>
<p>数据集中存在不同类型和比例的缺失值，需要采用多种方法进行处理和比较。</p>
<h2 id="项目目标" tabindex="-1"><a class="header-anchor" href="#项目目标"><span>项目目标</span></a></h2>
<ol>
<li>分析数据集中缺失值的模式和特征</li>
<li>实现并比较多种缺失值处理方法</li>
<li>评估不同缺失值处理方法对后续分析的影响</li>
<li>选择最佳的缺失值处理策略</li>
<li>准备完整的数据集用于疾病预测模型</li>
</ol>
<h2 id="实施步骤" tabindex="-1"><a class="header-anchor" href="#实施步骤"><span>实施步骤</span></a></h2>
<h3 id="步骤1-数据加载与缺失值分析" tabindex="-1"><a class="header-anchor" href="#步骤1-数据加载与缺失值分析"><span>步骤1：数据加载与缺失值分析</span></a></h3>
<p>首先，我们加载数据并分析缺失值的模式和特征。</p>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd</span>
<span class="line"><span class="token keyword">import</span> numpy <span class="token keyword">as</span> np</span>
<span class="line"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt</span>
<span class="line"><span class="token keyword">import</span> seaborn <span class="token keyword">as</span> sns</span>
<span class="line"><span class="token keyword">import</span> missingno <span class="token keyword">as</span> msno</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>impute <span class="token keyword">import</span> SimpleImputer<span class="token punctuation">,</span> KNNImputer<span class="token punctuation">,</span> IterativeImputer</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>ensemble <span class="token keyword">import</span> RandomForestRegressor</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>model_selection <span class="token keyword">import</span> train_test_split</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>metrics <span class="token keyword">import</span> mean_squared_error<span class="token punctuation">,</span> accuracy_score</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>preprocessing <span class="token keyword">import</span> StandardScaler</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 加载数据</span></span>
<span class="line">df <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'medical_data.csv'</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 查看数据基本信息</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>df<span class="token punctuation">.</span>info<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>df<span class="token punctuation">.</span>describe<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 分析缺失值</span></span>
<span class="line">missing <span class="token operator">=</span> df<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">missing_percent <span class="token operator">=</span> missing <span class="token operator">/</span> <span class="token builtin">len</span><span class="token punctuation">(</span>df<span class="token punctuation">)</span> <span class="token operator">*</span> <span class="token number">100</span></span>
<span class="line">missing_df <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span><span class="token punctuation">{</span><span class="token string">'missing_count'</span><span class="token punctuation">:</span> missing<span class="token punctuation">,</span> <span class="token string">'missing_percent'</span><span class="token punctuation">:</span> missing_percent<span class="token punctuation">}</span><span class="token punctuation">)</span></span>
<span class="line">missing_df <span class="token operator">=</span> missing_df<span class="token punctuation">[</span>missing_df<span class="token punctuation">[</span><span class="token string">'missing_count'</span><span class="token punctuation">]</span> <span class="token operator">></span> <span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">.</span>sort_values<span class="token punctuation">(</span><span class="token string">'missing_percent'</span><span class="token punctuation">,</span> ascending<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>missing_df<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 可视化缺失值模式</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">msno<span class="token punctuation">.</span>matrix<span class="token punctuation">(</span>df<span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'缺失值矩阵'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">msno<span class="token punctuation">.</span>heatmap<span class="token punctuation">(</span>df<span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'缺失值相关性热图'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 分析缺失值与目标变量的关系</span></span>
<span class="line"><span class="token comment"># 创建缺失标志列</span></span>
<span class="line"><span class="token keyword">for</span> col <span class="token keyword">in</span> df<span class="token punctuation">.</span>columns<span class="token punctuation">:</span></span>
<span class="line">    <span class="token keyword">if</span> df<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">></span> <span class="token number">0</span><span class="token punctuation">:</span></span>
<span class="line">        df<span class="token punctuation">[</span><span class="token string-interpolation"><span class="token string">f'</span><span class="token interpolation"><span class="token punctuation">{</span>col<span class="token punctuation">}</span></span><span class="token string">_missing'</span></span><span class="token punctuation">]</span> <span class="token operator">=</span> df<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>astype<span class="token punctuation">(</span><span class="token builtin">int</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 分析缺失标志与诊断结果的关系</span></span>
<span class="line"><span class="token keyword">for</span> col <span class="token keyword">in</span> <span class="token punctuation">[</span>c <span class="token keyword">for</span> c <span class="token keyword">in</span> df<span class="token punctuation">.</span>columns <span class="token keyword">if</span> c<span class="token punctuation">.</span>endswith<span class="token punctuation">(</span><span class="token string">'_missing'</span><span class="token punctuation">)</span><span class="token punctuation">]</span><span class="token punctuation">:</span></span>
<span class="line">    plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">8</span><span class="token punctuation">,</span> <span class="token number">5</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">    sns<span class="token punctuation">.</span>countplot<span class="token punctuation">(</span>x<span class="token operator">=</span>col<span class="token punctuation">,</span> hue<span class="token operator">=</span><span class="token string">'diagnosis'</span><span class="token punctuation">,</span> data<span class="token operator">=</span>df<span class="token punctuation">)</span></span>
<span class="line">    plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f'</span><span class="token interpolation"><span class="token punctuation">{</span>col<span class="token punctuation">}</span></span><span class="token string"> vs Diagnosis'</span></span><span class="token punctuation">)</span></span>
<span class="line">    plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="步骤2-准备数据用于缺失值处理比较" tabindex="-1"><a class="header-anchor" href="#步骤2-准备数据用于缺失值处理比较"><span>步骤2：准备数据用于缺失值处理比较</span></a></h3>
<p>为了比较不同缺失值处理方法的效果，我们需要准备一个完整的子集作为参考。</p>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># 选择完整记录的子集作为参考</span></span>
<span class="line">complete_cols <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">'age'</span><span class="token punctuation">,</span> <span class="token string">'gender'</span><span class="token punctuation">,</span> <span class="token string">'bmi'</span><span class="token punctuation">,</span> <span class="token string">'blood_pressure_systolic'</span><span class="token punctuation">,</span> <span class="token string">'blood_pressure_diastolic'</span><span class="token punctuation">,</span> </span>
<span class="line">                <span class="token string">'heart_rate'</span><span class="token punctuation">,</span> <span class="token string">'cholesterol'</span><span class="token punctuation">,</span> <span class="token string">'glucose'</span><span class="token punctuation">]</span></span>
<span class="line">complete_subset <span class="token operator">=</span> df<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span>subset<span class="token operator">=</span>complete_cols<span class="token punctuation">)</span><span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 从完整子集中随机引入缺失值，用于方法比较</span></span>
<span class="line">np<span class="token punctuation">.</span>random<span class="token punctuation">.</span>seed<span class="token punctuation">(</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">df_test <span class="token operator">=</span> complete_subset<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">mask <span class="token operator">=</span> np<span class="token punctuation">.</span>random<span class="token punctuation">.</span>rand<span class="token punctuation">(</span><span class="token operator">*</span>df_test<span class="token punctuation">[</span>complete_cols<span class="token punctuation">]</span><span class="token punctuation">.</span>shape<span class="token punctuation">)</span> <span class="token operator">&lt;</span> <span class="token number">0.2</span>  <span class="token comment"># 20%的缺失率</span></span>
<span class="line">df_test<span class="token punctuation">.</span>loc<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">,</span> complete_cols<span class="token punctuation">]</span> <span class="token operator">=</span> df_test<span class="token punctuation">[</span>complete_cols<span class="token punctuation">]</span><span class="token punctuation">.</span>mask<span class="token punctuation">(</span>mask<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 保存原始完整值用于评估</span></span>
<span class="line">true_values <span class="token operator">=</span> complete_subset<span class="token punctuation">[</span>complete_cols<span class="token punctuation">]</span><span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="步骤3-实现并比较不同的缺失值处理方法" tabindex="-1"><a class="header-anchor" href="#步骤3-实现并比较不同的缺失值处理方法"><span>步骤3：实现并比较不同的缺失值处理方法</span></a></h3>
<p>接下来，我们实现并比较多种缺失值处理方法。</p>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># 方法1：简单填充（均值/中位数/众数）</span></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">simple_imputation</span><span class="token punctuation">(</span>df<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">,</span> categorical_cols<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    df_imputed <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># 数值型特征使用中位数填充</span></span>
<span class="line">    <span class="token keyword">if</span> numeric_cols<span class="token punctuation">:</span></span>
<span class="line">        imputer <span class="token operator">=</span> SimpleImputer<span class="token punctuation">(</span>strategy<span class="token operator">=</span><span class="token string">'median'</span><span class="token punctuation">)</span></span>
<span class="line">        df_imputed<span class="token punctuation">[</span>numeric_cols<span class="token punctuation">]</span> <span class="token operator">=</span> imputer<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df<span class="token punctuation">[</span>numeric_cols<span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># 分类特征使用众数填充</span></span>
<span class="line">    <span class="token keyword">if</span> categorical_cols<span class="token punctuation">:</span></span>
<span class="line">        <span class="token keyword">for</span> col <span class="token keyword">in</span> categorical_cols<span class="token punctuation">:</span></span>
<span class="line">            df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span> <span class="token operator">=</span> df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>mode<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token keyword">return</span> df_imputed</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 方法2：KNN填充</span></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">knn_imputation</span><span class="token punctuation">(</span>df<span class="token punctuation">,</span> cols<span class="token punctuation">,</span> n_neighbors<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    df_imputed <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># 标准化数据</span></span>
<span class="line">    scaler <span class="token operator">=</span> StandardScaler<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    df_scaled <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span>scaler<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df<span class="token punctuation">[</span>cols<span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span> columns<span class="token operator">=</span>cols<span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># KNN填充</span></span>
<span class="line">    imputer <span class="token operator">=</span> KNNImputer<span class="token punctuation">(</span>n_neighbors<span class="token operator">=</span>n_neighbors<span class="token punctuation">)</span></span>
<span class="line">    df_imputed<span class="token punctuation">[</span>cols<span class="token punctuation">]</span> <span class="token operator">=</span> imputer<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df_scaled<span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># 反标准化</span></span>
<span class="line">    df_imputed<span class="token punctuation">[</span>cols<span class="token punctuation">]</span> <span class="token operator">=</span> scaler<span class="token punctuation">.</span>inverse_transform<span class="token punctuation">(</span>df_imputed<span class="token punctuation">[</span>cols<span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token keyword">return</span> df_imputed</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 方法3：多重插补（使用IterativeImputer）</span></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">iterative_imputation</span><span class="token punctuation">(</span>df<span class="token punctuation">,</span> cols<span class="token punctuation">,</span> max_iter<span class="token operator">=</span><span class="token number">10</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    df_imputed <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># 使用随机森林作为估计器</span></span>
<span class="line">    estimator <span class="token operator">=</span> RandomForestRegressor<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span>random_state<span class="token punctuation">)</span></span>
<span class="line">    imputer <span class="token operator">=</span> IterativeImputer<span class="token punctuation">(</span>estimator<span class="token operator">=</span>estimator<span class="token punctuation">,</span> max_iter<span class="token operator">=</span>max_iter<span class="token punctuation">,</span> random_state<span class="token operator">=</span>random_state<span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    df_imputed<span class="token punctuation">[</span>cols<span class="token punctuation">]</span> <span class="token operator">=</span> imputer<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df<span class="token punctuation">[</span>cols<span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token keyword">return</span> df_imputed</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 方法4：基于分组的填充</span></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">group_imputation</span><span class="token punctuation">(</span>df<span class="token punctuation">,</span> target_cols<span class="token punctuation">,</span> group_cols<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    df_imputed <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token keyword">for</span> col <span class="token keyword">in</span> target_cols<span class="token punctuation">:</span></span>
<span class="line">        <span class="token comment"># 对每个分组计算中位数</span></span>
<span class="line">        group_medians <span class="token operator">=</span> df<span class="token punctuation">.</span>groupby<span class="token punctuation">(</span>group_cols<span class="token punctuation">)</span><span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>transform<span class="token punctuation">(</span><span class="token string">'median'</span><span class="token punctuation">)</span></span>
<span class="line">        <span class="token comment"># 使用分组中位数填充</span></span>
<span class="line">        df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span> <span class="token operator">=</span> df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>group_medians<span class="token punctuation">)</span></span>
<span class="line">        <span class="token comment"># 如果仍有缺失值（可能是整个组都缺失），使用全局中位数填充</span></span>
<span class="line">        df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span> <span class="token operator">=</span> df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>median<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token keyword">return</span> df_imputed</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 应用不同的填充方法</span></span>
<span class="line">numeric_cols <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">'age'</span><span class="token punctuation">,</span> <span class="token string">'bmi'</span><span class="token punctuation">,</span> <span class="token string">'blood_pressure_systolic'</span><span class="token punctuation">,</span> <span class="token string">'blood_pressure_diastolic'</span><span class="token punctuation">,</span> </span>
<span class="line">               <span class="token string">'heart_rate'</span><span class="token punctuation">,</span> <span class="token string">'cholesterol'</span><span class="token punctuation">,</span> <span class="token string">'glucose'</span><span class="token punctuation">]</span></span>
<span class="line">categorical_cols <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">'gender'</span><span class="token punctuation">,</span> <span class="token string">'smoking'</span><span class="token punctuation">,</span> <span class="token string">'alcohol_consumption'</span><span class="token punctuation">,</span> <span class="token string">'physical_activity'</span><span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 应用各种方法</span></span>
<span class="line">df_simple <span class="token operator">=</span> simple_imputation<span class="token punctuation">(</span>df_test<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">,</span> categorical_cols<span class="token punctuation">)</span></span>
<span class="line">df_knn <span class="token operator">=</span> knn_imputation<span class="token punctuation">(</span>df_test<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line">df_iterative <span class="token operator">=</span> iterative_imputation<span class="token punctuation">(</span>df_test<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line">df_group <span class="token operator">=</span> group_imputation<span class="token punctuation">(</span>df_test<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">,</span> <span class="token punctuation">[</span><span class="token string">'gender'</span><span class="token punctuation">,</span> <span class="token string">'age'</span><span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 评估不同方法的性能</span></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">evaluate_imputation</span><span class="token punctuation">(</span>imputed_df<span class="token punctuation">,</span> true_df<span class="token punctuation">,</span> cols<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    results <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token punctuation">}</span></span>
<span class="line">    <span class="token keyword">for</span> col <span class="token keyword">in</span> cols<span class="token punctuation">:</span></span>
<span class="line">        <span class="token comment"># 只考虑原本缺失的值</span></span>
<span class="line">        mask <span class="token operator">=</span> imputed_df<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>notnull<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">&amp;</span> df_test<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">        <span class="token keyword">if</span> mask<span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">></span> <span class="token number">0</span><span class="token punctuation">:</span></span>
<span class="line">            mse <span class="token operator">=</span> mean_squared_error<span class="token punctuation">(</span>true_df<span class="token punctuation">.</span>loc<span class="token punctuation">[</span>mask<span class="token punctuation">,</span> col<span class="token punctuation">]</span><span class="token punctuation">,</span> imputed_df<span class="token punctuation">.</span>loc<span class="token punctuation">[</span>mask<span class="token punctuation">,</span> col<span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">            results<span class="token punctuation">[</span>col<span class="token punctuation">]</span> <span class="token operator">=</span> mse</span>
<span class="line">    <span class="token keyword">return</span> results</span>
<span class="line"></span>
<span class="line"><span class="token comment"># 评估各种方法</span></span>
<span class="line">simple_results <span class="token operator">=</span> evaluate_imputation<span class="token punctuation">(</span>df_simple<span class="token punctuation">,</span> true_values<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line">knn_results <span class="token operator">=</span> evaluate_imputation<span class="token punctuation">(</span>df_knn<span class="token punctuation">,</span> true_values<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line">iterative_results <span class="token operator">=</span> evaluate_imputation<span class="token punctuation">(</span>df_iterative<span class="token punctuation">,</span> true_values<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line">group_results <span class="token operator">=</span> evaluate_imputation<span class="token punctuation">(</span>df_group<span class="token punctuation">,</span> true_values<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 比较结果</span></span>
<span class="line">results_df <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span><span class="token punctuation">{</span></span>
<span class="line">    <span class="token string">'Simple'</span><span class="token punctuation">:</span> simple_results<span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'KNN'</span><span class="token punctuation">:</span> knn_results<span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Iterative'</span><span class="token punctuation">:</span> iterative_results<span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Group'</span><span class="token punctuation">:</span> group_results</span>
<span class="line"><span class="token punctuation">}</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>results_df<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 可视化比较</span></span>
<span class="line">results_df<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>kind<span class="token operator">=</span><span class="token string">'bar'</span><span class="token punctuation">,</span> figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'不同填充方法的MSE比较'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'均方误差 (MSE)'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xticks<span class="token punctuation">(</span>rotation<span class="token operator">=</span><span class="token number">45</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>tight_layout<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="步骤4-选择最佳方法并处理完整数据集" tabindex="-1"><a class="header-anchor" href="#步骤4-选择最佳方法并处理完整数据集"><span>步骤4：选择最佳方法并处理完整数据集</span></a></h3>
<p>根据比较结果，我们选择最佳的缺失值处理方法，并应用于完整数据集。</p>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># 假设迭代填充方法表现最好</span></span>
<span class="line">best_method <span class="token operator">=</span> <span class="token string">'Iterative'</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"选择 </span><span class="token interpolation"><span class="token punctuation">{</span>best_method<span class="token punctuation">}</span></span><span class="token string"> 作为最佳填充方法"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 处理完整数据集</span></span>
<span class="line"><span class="token comment"># 首先处理数值型特征</span></span>
<span class="line">df_complete <span class="token operator">=</span> iterative_imputation<span class="token punctuation">(</span>df<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 然后处理分类特征</span></span>
<span class="line"><span class="token keyword">for</span> col <span class="token keyword">in</span> categorical_cols<span class="token punctuation">:</span></span>
<span class="line">    <span class="token keyword">if</span> df_complete<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">></span> <span class="token number">0</span><span class="token punctuation">:</span></span>
<span class="line">        df_complete<span class="token punctuation">[</span>col<span class="token punctuation">]</span> <span class="token operator">=</span> df_complete<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_complete<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>mode<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 检查处理后的缺失值情况</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"处理后的缺失值情况:"</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>df_complete<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 保存处理后的数据集</span></span>
<span class="line">df_complete<span class="token punctuation">.</span>to_csv<span class="token punctuation">(</span><span class="token string">'medical_data_complete.csv'</span><span class="token punctuation">,</span> index<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="步骤5-评估缺失值处理对后续分析的影响" tabindex="-1"><a class="header-anchor" href="#步骤5-评估缺失值处理对后续分析的影响"><span>步骤5：评估缺失值处理对后续分析的影响</span></a></h3>
<p>最后，我们评估缺失值处理对疾病预测模型的影响。</p>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># 准备用于预测的特征和目标变量</span></span>
<span class="line">X <span class="token operator">=</span> df_complete<span class="token punctuation">.</span>drop<span class="token punctuation">(</span><span class="token punctuation">[</span><span class="token string">'patient_id'</span><span class="token punctuation">,</span> <span class="token string">'diagnosis'</span><span class="token punctuation">]</span> <span class="token operator">+</span> </span>
<span class="line">                    <span class="token punctuation">[</span>c <span class="token keyword">for</span> c <span class="token keyword">in</span> df_complete<span class="token punctuation">.</span>columns <span class="token keyword">if</span> c<span class="token punctuation">.</span>endswith<span class="token punctuation">(</span><span class="token string">'_missing'</span><span class="token punctuation">)</span><span class="token punctuation">]</span><span class="token punctuation">,</span> axis<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span></span>
<span class="line">y <span class="token operator">=</span> df_complete<span class="token punctuation">[</span><span class="token string">'diagnosis'</span><span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 将分类特征转换为数值</span></span>
<span class="line">X <span class="token operator">=</span> pd<span class="token punctuation">.</span>get_dummies<span class="token punctuation">(</span>X<span class="token punctuation">,</span> drop_first<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 分割训练集和测试集</span></span>
<span class="line">X_train<span class="token punctuation">,</span> X_test<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> y_test <span class="token operator">=</span> train_test_split<span class="token punctuation">(</span>X<span class="token punctuation">,</span> y<span class="token punctuation">,</span> test_size<span class="token operator">=</span><span class="token number">0.2</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 训练随机森林分类器</span></span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>ensemble <span class="token keyword">import</span> RandomForestClassifier</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>metrics <span class="token keyword">import</span> classification_report<span class="token punctuation">,</span> confusion_matrix</span>
<span class="line"></span>
<span class="line">clf <span class="token operator">=</span> RandomForestClassifier<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">clf<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train<span class="token punctuation">,</span> y_train<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 预测和评估</span></span>
<span class="line">y_pred <span class="token operator">=</span> clf<span class="token punctuation">.</span>predict<span class="token punctuation">(</span>X_test<span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"分类报告:"</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>classification_report<span class="token punctuation">(</span>y_test<span class="token punctuation">,</span> y_pred<span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 可视化混淆矩阵</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">8</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>heatmap<span class="token punctuation">(</span>confusion_matrix<span class="token punctuation">(</span>y_test<span class="token punctuation">,</span> y_pred<span class="token punctuation">)</span><span class="token punctuation">,</span> annot<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span> fmt<span class="token operator">=</span><span class="token string">'d'</span><span class="token punctuation">,</span> cmap<span class="token operator">=</span><span class="token string">'Blues'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'预测标签'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'真实标签'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'混淆矩阵'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># 特征重要性</span></span>
<span class="line">feature_importance <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span><span class="token punctuation">{</span></span>
<span class="line">    <span class="token string">'feature'</span><span class="token punctuation">:</span> X<span class="token punctuation">.</span>columns<span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'importance'</span><span class="token punctuation">:</span> clf<span class="token punctuation">.</span>feature_importances_</span>
<span class="line"><span class="token punctuation">}</span><span class="token punctuation">)</span><span class="token punctuation">.</span>sort_values<span class="token punctuation">(</span><span class="token string">'importance'</span><span class="token punctuation">,</span> ascending<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>barplot<span class="token punctuation">(</span>x<span class="token operator">=</span><span class="token string">'importance'</span><span class="token punctuation">,</span> y<span class="token operator">=</span><span class="token string">'feature'</span><span class="token punctuation">,</span> data<span class="token operator">=</span>feature_importance<span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token number">15</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'特征重要性'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>tight_layout<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="结果分析" tabindex="-1"><a class="header-anchor" href="#结果分析"><span>结果分析</span></a></h2>
<p>通过比较不同的缺失值处理方法，我们得出以下结论：</p>
<ol>
<li><strong>迭代填充方法</strong>在大多数特征上表现最好，特别是对于相关性高的特征</li>
<li><strong>KNN填充</strong>在某些特征上表现良好，但计算成本较高</li>
<li><strong>分组填充</strong>对于与分组变量强相关的特征效果好</li>
<li><strong>简单填充</strong>虽然简单，但在大多数情况下精度较低</li>
</ol>
<p>最终选择的迭代填充方法成功处理了数据集中的缺失值，为后续的疾病预测模型提供了高质量的输入数据。预测模型在测试集上取得了良好的性能，表明我们的缺失值处理策略是有效的。</p>
<h2 id="进阶挑战" tabindex="-1"><a class="header-anchor" href="#进阶挑战"><span>进阶挑战</span></a></h2>
<p>如果你已经完成了基本任务，可以尝试以下进阶挑战：</p>
<ol>
<li><strong>缺失机制分析</strong>：深入研究数据的缺失机制（MCAR、MAR、MNAR）</li>
<li><strong>敏感性分析</strong>：评估不同缺失值处理方法对最终模型结果的敏感性</li>
<li><strong>多重插补进阶</strong>：实现完整的多重插补流程，包括创建多个填充数据集和合并结果</li>
<li><strong>自定义填充模型</strong>：为特定特征开发定制的预测模型用于填充</li>
<li><strong>缺失值模拟</strong>：设计实验，通过在完整数据上模拟不同模式的缺失值，评估各种方法的鲁棒性</li>
</ol>
<h2 id="小结与反思" tabindex="-1"><a class="header-anchor" href="#小结与反思"><span>小结与反思</span></a></h2>
<p>通过这个项目，我们学习了如何处理医疗数据中的缺失值，并比较了不同方法的效果。缺失值处理是医疗数据分析中的关键步骤，直接影响后续分析和预测的准确性。</p>
<p>在实际应用中，这类缺失值处理技术可以帮助医疗机构更好地利用不完整的患者数据，提高疾病预测和诊断的准确性。例如，通过适当的缺失值处理，即使某些检查结果缺失，也能为患者提供相对准确的风险评估。</p>
<h3 id="思考问题" tabindex="-1"><a class="header-anchor" href="#思考问题"><span>思考问题</span></a></h3>
<ol>
<li>在医疗数据中，缺失值本身可能包含信息（例如，医生没有要求某项检查）。如何在填充缺失值的同时保留这种信息？</li>
<li>不同类型的医疗数据（如实验室检查、问卷调查、影像数据）可能需要不同的缺失值处理策略。如何为不同类型的数据选择合适的方法？</li>
<li>在处理敏感的医疗数据时，如何平衡数据完整性和隐私保护的需求？</li>
</ol>
<div class="practice-link">
  <a href="/projects/classification/titanic.html" class="button">下一个模块：分类算法项目</a>
</div> </div></template>


