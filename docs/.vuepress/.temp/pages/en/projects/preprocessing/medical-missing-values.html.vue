<template><div><h1 id="handling-missing-values-in-medical-data" tabindex="-1"><a class="header-anchor" href="#handling-missing-values-in-medical-data"><span>Handling Missing Values in Medical Data</span></a></h1>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span>Project Overview
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li><strong>Difficulty</strong>: Intermediate</li>
      <li><strong>Type</strong>: Missing Value Handling</li>
      <!-- <li><strong>Estimated Time</strong>: 5-7 hours</li> -->
      <li><strong>Skills</strong>: Missing Pattern Analysis, Multiple Imputation, KNN Imputation, Model-based Imputation</li>
      <li><strong>Related Knowledge Module</strong>: <a href="/en/core/preprocessing/data-presentation.html">Data Preprocessing</a></li>
    </ul>
  </div>
</div>
<h2 id="project-background" tabindex="-1"><a class="header-anchor" href="#project-background"><span>Project Background</span></a></h2>
<p>Medical data plays a key role in clinical research, disease prediction, and optimizing treatment plans. However, medical datasets often contain a large amount of missing values, which can be caused by factors such as equipment malfunction, patients not completing all tests, recording errors, or data entry issues. Improper handling of missing values may lead to biased conclusions and affect the accuracy of medical decisions.</p>
<p>In this project, we will work on a medical dataset with various types of missing values, compare the effects of different missing value handling methods, and prepare high-quality data for subsequent disease prediction models.</p>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span>Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>Missing values in medical data are often not random. For example, certain tests might only be conducted on patients with specific symptoms, resulting in missing data that is related to the patient’s health status. This non-random missing pattern requires special treatment to avoid introducing bias.</p>
  </div>
</div>
<h2 id="dataset-introduction" tabindex="-1"><a class="header-anchor" href="#dataset-introduction"><span>Dataset Introduction</span></a></h2>
<p>The dataset used in this project contains medical records of 5,000 patients, including the following fields:</p>
<ul>
<li><strong>patient_id</strong>: Patient ID</li>
<li><strong>age</strong>: Age</li>
<li><strong>gender</strong>: Gender</li>
<li><strong>bmi</strong>: Body Mass Index</li>
<li><strong>blood_pressure_systolic</strong>: Systolic Blood Pressure</li>
<li><strong>blood_pressure_diastolic</strong>: Diastolic Blood Pressure</li>
<li><strong>heart_rate</strong>: Heart Rate</li>
<li><strong>cholesterol</strong>: Cholesterol Level</li>
<li><strong>glucose</strong>: Glucose Level</li>
<li><strong>smoking</strong>: Smoking Status</li>
<li><strong>alcohol_consumption</strong>: Alcohol Consumption Level</li>
<li><strong>physical_activity</strong>: Physical Activity Level</li>
<li><strong>family_history</strong>: Family History</li>
<li><strong>medication</strong>: Current Medication</li>
<li><strong>diagnosis</strong>: Diagnosis Result</li>
</ul>
<p>The dataset has different types and proportions of missing values, which require multiple methods for handling and comparison.</p>
<h2 id="project-objectives" tabindex="-1"><a class="header-anchor" href="#project-objectives"><span>Project Objectives</span></a></h2>
<ol>
<li>Analyze the patterns and characteristics of missing values in the dataset.</li>
<li>Implement and compare various missing value handling methods.</li>
<li>Evaluate the impact of different missing value handling methods on subsequent analysis.</li>
<li>Select the best missing value handling strategy.</li>
<li>Prepare a complete dataset for disease prediction models.</li>
</ol>
<h2 id="implementation-steps" tabindex="-1"><a class="header-anchor" href="#implementation-steps"><span>Implementation Steps</span></a></h2>
<h3 id="step-1-data-loading-and-missing-value-analysis" tabindex="-1"><a class="header-anchor" href="#step-1-data-loading-and-missing-value-analysis"><span>Step 1: Data Loading and Missing Value Analysis</span></a></h3>
<p>First, we load the data and analyze the patterns and characteristics of the missing values.</p>
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
<span class="line"><span class="token comment"># Load the data</span></span>
<span class="line">df <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'medical_data.csv'</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># View basic information about the data</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>df<span class="token punctuation">.</span>info<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>df<span class="token punctuation">.</span>describe<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Analyze missing values</span></span>
<span class="line">missing <span class="token operator">=</span> df<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">missing_percent <span class="token operator">=</span> missing <span class="token operator">/</span> <span class="token builtin">len</span><span class="token punctuation">(</span>df<span class="token punctuation">)</span> <span class="token operator">*</span> <span class="token number">100</span></span>
<span class="line">missing_df <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span><span class="token punctuation">{</span><span class="token string">'missing_count'</span><span class="token punctuation">:</span> missing<span class="token punctuation">,</span> <span class="token string">'missing_percent'</span><span class="token punctuation">:</span> missing_percent<span class="token punctuation">}</span><span class="token punctuation">)</span></span>
<span class="line">missing_df <span class="token operator">=</span> missing_df<span class="token punctuation">[</span>missing_df<span class="token punctuation">[</span><span class="token string">'missing_count'</span><span class="token punctuation">]</span> <span class="token operator">></span> <span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">.</span>sort_values<span class="token punctuation">(</span><span class="token string">'missing_percent'</span><span class="token punctuation">,</span> ascending<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>missing_df<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Visualize missing value patterns</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">msno<span class="token punctuation">.</span>matrix<span class="token punctuation">(</span>df<span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Missing Value Matrix'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">msno<span class="token punctuation">.</span>heatmap<span class="token punctuation">(</span>df<span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Missing Value Correlation Heatmap'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Analyze the relationship between missing values and the target variable</span></span>
<span class="line"><span class="token comment"># Create missing indicator columns</span></span>
<span class="line"><span class="token keyword">for</span> col <span class="token keyword">in</span> df<span class="token punctuation">.</span>columns<span class="token punctuation">:</span></span>
<span class="line">    <span class="token keyword">if</span> df<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">></span> <span class="token number">0</span><span class="token punctuation">:</span></span>
<span class="line">        df<span class="token punctuation">[</span><span class="token string-interpolation"><span class="token string">f'</span><span class="token interpolation"><span class="token punctuation">{</span>col<span class="token punctuation">}</span></span><span class="token string">_missing'</span></span><span class="token punctuation">]</span> <span class="token operator">=</span> df<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>astype<span class="token punctuation">(</span><span class="token builtin">int</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Analyze the relationship between missing indicators and diagnosis results</span></span>
<span class="line"><span class="token keyword">for</span> col <span class="token keyword">in</span> <span class="token punctuation">[</span>c <span class="token keyword">for</span> c <span class="token keyword">in</span> df<span class="token punctuation">.</span>columns <span class="token keyword">if</span> c<span class="token punctuation">.</span>endswith<span class="token punctuation">(</span><span class="token string">'_missing'</span><span class="token punctuation">)</span><span class="token punctuation">]</span><span class="token punctuation">:</span></span>
<span class="line">    plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">8</span><span class="token punctuation">,</span> <span class="token number">5</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">    sns<span class="token punctuation">.</span>countplot<span class="token punctuation">(</span>x<span class="token operator">=</span>col<span class="token punctuation">,</span> hue<span class="token operator">=</span><span class="token string">'diagnosis'</span><span class="token punctuation">,</span> data<span class="token operator">=</span>df<span class="token punctuation">)</span></span>
<span class="line">    plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f'</span><span class="token interpolation"><span class="token punctuation">{</span>col<span class="token punctuation">}</span></span><span class="token string"> vs Diagnosis'</span></span><span class="token punctuation">)</span></span>
<span class="line">    plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="step-2-prepare-data-for-comparing-missing-value-handling-methods" tabindex="-1"><a class="header-anchor" href="#step-2-prepare-data-for-comparing-missing-value-handling-methods"><span>Step 2: Prepare Data for Comparing Missing Value Handling Methods</span></a></h3>
<p>To compare the effects of different missing value handling methods, we need to prepare a complete subset as a reference.</p>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># Select a subset with complete records as a reference</span></span>
<span class="line">complete_cols <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">'age'</span><span class="token punctuation">,</span> <span class="token string">'gender'</span><span class="token punctuation">,</span> <span class="token string">'bmi'</span><span class="token punctuation">,</span> <span class="token string">'blood_pressure_systolic'</span><span class="token punctuation">,</span> <span class="token string">'blood_pressure_diastolic'</span><span class="token punctuation">,</span> </span>
<span class="line">                <span class="token string">'heart_rate'</span><span class="token punctuation">,</span> <span class="token string">'cholesterol'</span><span class="token punctuation">,</span> <span class="token string">'glucose'</span><span class="token punctuation">]</span></span>
<span class="line">complete_subset <span class="token operator">=</span> df<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span>subset<span class="token operator">=</span>complete_cols<span class="token punctuation">)</span><span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Randomly introduce missing values in the complete subset for method comparison</span></span>
<span class="line">np<span class="token punctuation">.</span>random<span class="token punctuation">.</span>seed<span class="token punctuation">(</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">df_test <span class="token operator">=</span> complete_subset<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">mask <span class="token operator">=</span> np<span class="token punctuation">.</span>random<span class="token punctuation">.</span>rand<span class="token punctuation">(</span><span class="token operator">*</span>df_test<span class="token punctuation">[</span>complete_cols<span class="token punctuation">]</span><span class="token punctuation">.</span>shape<span class="token punctuation">)</span> <span class="token operator">&lt;</span> <span class="token number">0.2</span>  <span class="token comment"># 20% missing rate</span></span>
<span class="line">df_test<span class="token punctuation">.</span>loc<span class="token punctuation">[</span><span class="token punctuation">:</span><span class="token punctuation">,</span> complete_cols<span class="token punctuation">]</span> <span class="token operator">=</span> df_test<span class="token punctuation">[</span>complete_cols<span class="token punctuation">]</span><span class="token punctuation">.</span>mask<span class="token punctuation">(</span>mask<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Save the original complete values for evaluation</span></span>
<span class="line">true_values <span class="token operator">=</span> complete_subset<span class="token punctuation">[</span>complete_cols<span class="token punctuation">]</span><span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="step-3-implement-and-compare-different-missing-value-handling-methods" tabindex="-1"><a class="header-anchor" href="#step-3-implement-and-compare-different-missing-value-handling-methods"><span>Step 3: Implement and Compare Different Missing Value Handling Methods</span></a></h3>
<p>Next, we implement and compare several methods for handling missing values.</p>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># Method 1: Simple Imputation (mean/median/mode)</span></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">simple_imputation</span><span class="token punctuation">(</span>df<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">,</span> categorical_cols<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    df_imputed <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># Use median for numerical features</span></span>
<span class="line">    <span class="token keyword">if</span> numeric_cols<span class="token punctuation">:</span></span>
<span class="line">        imputer <span class="token operator">=</span> SimpleImputer<span class="token punctuation">(</span>strategy<span class="token operator">=</span><span class="token string">'median'</span><span class="token punctuation">)</span></span>
<span class="line">        df_imputed<span class="token punctuation">[</span>numeric_cols<span class="token punctuation">]</span> <span class="token operator">=</span> imputer<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df<span class="token punctuation">[</span>numeric_cols<span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># Use mode for categorical features</span></span>
<span class="line">    <span class="token keyword">if</span> categorical_cols<span class="token punctuation">:</span></span>
<span class="line">        <span class="token keyword">for</span> col <span class="token keyword">in</span> categorical_cols<span class="token punctuation">:</span></span>
<span class="line">            df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span> <span class="token operator">=</span> df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>mode<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token keyword">return</span> df_imputed</span>
<span class="line"></span>
<span class="line"><span class="token comment"># Method 2: KNN Imputation</span></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">knn_imputation</span><span class="token punctuation">(</span>df<span class="token punctuation">,</span> cols<span class="token punctuation">,</span> n_neighbors<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    df_imputed <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># Standardize the data</span></span>
<span class="line">    scaler <span class="token operator">=</span> StandardScaler<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    df_scaled <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span>scaler<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df<span class="token punctuation">[</span>cols<span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span> columns<span class="token operator">=</span>cols<span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># KNN imputation</span></span>
<span class="line">    imputer <span class="token operator">=</span> KNNImputer<span class="token punctuation">(</span>n_neighbors<span class="token operator">=</span>n_neighbors<span class="token punctuation">)</span></span>
<span class="line">    df_imputed<span class="token punctuation">[</span>cols<span class="token punctuation">]</span> <span class="token operator">=</span> imputer<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df_scaled<span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># Reverse standardization</span></span>
<span class="line">    df_imputed<span class="token punctuation">[</span>cols<span class="token punctuation">]</span> <span class="token operator">=</span> scaler<span class="token punctuation">.</span>inverse_transform<span class="token punctuation">(</span>df_imputed<span class="token punctuation">[</span>cols<span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token keyword">return</span> df_imputed</span>
<span class="line"></span>
<span class="line"><span class="token comment"># Method 3: Multiple Imputation (using IterativeImputer)</span></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">iterative_imputation</span><span class="token punctuation">(</span>df<span class="token punctuation">,</span> cols<span class="token punctuation">,</span> max_iter<span class="token operator">=</span><span class="token number">10</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">0</span><span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    df_imputed <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment"># Use RandomForest as the estimator</span></span>
<span class="line">    estimator <span class="token operator">=</span> RandomForestRegressor<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span>random_state<span class="token punctuation">)</span></span>
<span class="line">    imputer <span class="token operator">=</span> IterativeImputer<span class="token punctuation">(</span>estimator<span class="token operator">=</span>estimator<span class="token punctuation">,</span> max_iter<span class="token operator">=</span>max_iter<span class="token punctuation">,</span> random_state<span class="token operator">=</span>random_state<span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    df_imputed<span class="token punctuation">[</span>cols<span class="token punctuation">]</span> <span class="token operator">=</span> imputer<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df<span class="token punctuation">[</span>cols<span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token keyword">return</span> df_imputed</span>
<span class="line"></span>
<span class="line"><span class="token comment"># Method 4: Group-based Imputation</span></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">group_imputation</span><span class="token punctuation">(</span>df<span class="token punctuation">,</span> target_cols<span class="token punctuation">,</span> group_cols<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    df_imputed <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token keyword">for</span> col <span class="token keyword">in</span> target_cols<span class="token punctuation">:</span></span>
<span class="line">        <span class="token comment"># Calculate the median for each group</span></span>
<span class="line">        group_medians <span class="token operator">=</span> df<span class="token punctuation">.</span>groupby<span class="token punctuation">(</span>group_cols<span class="token punctuation">)</span><span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>transform<span class="token punctuation">(</span><span class="token string">'median'</span><span class="token punctuation">)</span></span>
<span class="line">        <span class="token comment"># Fill missing values with group median</span></span>
<span class="line">        df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span> <span class="token operator">=</span> df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>group_medians<span class="token punctuation">)</span></span>
<span class="line">        <span class="token comment"># If there are still missing values (e.g., the entire group is missing), fill with the global median</span></span>
<span class="line">        df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span> <span class="token operator">=</span> df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_imputed<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>median<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">    </span>
<span class="line">    <span class="token keyword">return</span> df_imputed</span>
<span class="line"></span>
<span class="line"><span class="token comment"># Apply different imputation methods</span></span>
<span class="line">numeric_cols <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">'age'</span><span class="token punctuation">,</span> <span class="token string">'bmi'</span><span class="token punctuation">,</span> <span class="token string">'blood_pressure_systolic'</span><span class="token punctuation">,</span> <span class="token string">'blood_pressure_diastolic'</span><span class="token punctuation">,</span> </span>
<span class="line">               <span class="token string">'heart_rate'</span><span class="token punctuation">,</span> <span class="token string">'cholesterol'</span><span class="token punctuation">,</span> <span class="token string">'glucose'</span><span class="token punctuation">]</span></span>
<span class="line">categorical_cols <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token string">'gender'</span><span class="token punctuation">,</span> <span class="token string">'smoking'</span><span class="token punctuation">,</span> <span class="token string">'alcohol_consumption'</span><span class="token punctuation">,</span> <span class="token string">'physical_activity'</span><span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Apply various methods</span></span>
<span class="line">df_simple <span class="token operator">=</span> simple_imputation<span class="token punctuation">(</span>df_test<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">,</span> categorical_cols<span class="token punctuation">)</span></span>
<span class="line">df_knn <span class="token operator">=</span> knn_imputation<span class="token punctuation">(</span>df_test<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line">df_iterative <span class="token operator">=</span> iterative_imputation<span class="token punctuation">(</span>df_test<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line">df_group <span class="token operator">=</span> group_imputation<span class="token punctuation">(</span>df_test<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">,</span> <span class="token punctuation">[</span><span class="token string">'gender'</span><span class="token punctuation">,</span> <span class="token string">'age'</span><span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Evaluate the performance of different methods</span></span>
<span class="line"><span class="token keyword">def</span> <span class="token function">evaluate_imputation</span><span class="token punctuation">(</span>imputed_df<span class="token punctuation">,</span> true_df<span class="token punctuation">,</span> cols<span class="token punctuation">)</span><span class="token punctuation">:</span></span>
<span class="line">    results <span class="token operator">=</span> <span class="token punctuation">{</span><span class="token punctuation">}</span></span>
<span class="line">    <span class="token keyword">for</span> col <span class="token keyword">in</span> cols<span class="token punctuation">:</span></span>
<span class="line">        <span class="token comment"># Only consider the originally missing values</span></span>
<span class="line">        mask <span class="token operator">=</span> imputed_df<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>notnull<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">&amp;</span> df_test<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">        <span class="token keyword">if</span> mask<span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">></span> <span class="token number">0</span><span class="token punctuation">:</span></span>
<span class="line">            mse <span class="token operator">=</span> mean_squared_error<span class="token punctuation">(</span>true_df<span class="token punctuation">.</span>loc<span class="token punctuation">[</span>mask<span class="token punctuation">,</span> col<span class="token punctuation">]</span><span class="token punctuation">,</span> imputed_df<span class="token punctuation">.</span>loc<span class="token punctuation">[</span>mask<span class="token punctuation">,</span> col<span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line">            results<span class="token punctuation">[</span>col<span class="token punctuation">]</span> <span class="token operator">=</span> mse</span>
<span class="line">    <span class="token keyword">return</span> results</span>
<span class="line"></span>
<span class="line"><span class="token comment"># Evaluate various methods</span></span>
<span class="line">simple_results <span class="token operator">=</span> evaluate_imputation<span class="token punctuation">(</span>df_simple<span class="token punctuation">,</span> true_values<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line">knn_results <span class="token operator">=</span> evaluate_imputation<span class="token punctuation">(</span>df_knn<span class="token punctuation">,</span> true_values<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line">iterative_results <span class="token operator">=</span> evaluate_imputation<span class="token punctuation">(</span>df_iterative<span class="token punctuation">,</span> true_values<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line">group_results <span class="token operator">=</span> evaluate_imputation<span class="token punctuation">(</span>df_group<span class="token punctuation">,</span> true_values<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Compare the results</span></span>
<span class="line">results_df <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span><span class="token punctuation">{</span></span>
<span class="line">    <span class="token string">'Simple'</span><span class="token punctuation">:</span> simple_results<span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'KNN'</span><span class="token punctuation">:</span> knn_results<span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Iterative'</span><span class="token punctuation">:</span> iterative_results<span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'Group'</span><span class="token punctuation">:</span> group_results</span>
<span class="line"><span class="token punctuation">}</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>results_df<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Visualize the comparison</span></span>
<span class="line">results_df<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>kind<span class="token operator">=</span><span class="token string">'bar'</span><span class="token punctuation">,</span> figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'MSE Comparison of Different Imputation Methods'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'Mean Squared Error (MSE)'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xticks<span class="token punctuation">(</span>rotation<span class="token operator">=</span><span class="token number">45</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>tight_layout<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="step-4-select-the-best-method-and-process-the-complete-dataset" tabindex="-1"><a class="header-anchor" href="#step-4-select-the-best-method-and-process-the-complete-dataset"><span>Step 4: Select the Best Method and Process the Complete Dataset</span></a></h3>
<p>Based on the comparison results, we choose the best missing value handling method and apply it to the complete dataset.</p>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># Assuming the iterative imputation method performed the best</span></span>
<span class="line">best_method <span class="token operator">=</span> <span class="token string">'Iterative'</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"Selected </span><span class="token interpolation"><span class="token punctuation">{</span>best_method<span class="token punctuation">}</span></span><span class="token string"> as the best imputation method"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Process the complete dataset</span></span>
<span class="line"><span class="token comment"># First, handle numerical features</span></span>
<span class="line">df_complete <span class="token operator">=</span> iterative_imputation<span class="token punctuation">(</span>df<span class="token punctuation">,</span> numeric_cols<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Then, handle categorical features</span></span>
<span class="line"><span class="token keyword">for</span> col <span class="token keyword">in</span> categorical_cols<span class="token punctuation">:</span></span>
<span class="line">    <span class="token keyword">if</span> df_complete<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">></span> <span class="token number">0</span><span class="token punctuation">:</span></span>
<span class="line">        df_complete<span class="token punctuation">[</span>col<span class="token punctuation">]</span> <span class="token operator">=</span> df_complete<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_complete<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>mode<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Check the missing value status after processing</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Missing values after processing:"</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>df_complete<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Save the processed dataset</span></span>
<span class="line">df_complete<span class="token punctuation">.</span>to_csv<span class="token punctuation">(</span><span class="token string">'medical_data_complete.csv'</span><span class="token punctuation">,</span> index<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h3 id="step-5-evaluate-the-impact-of-missing-value-handling-on-subsequent-analysis" tabindex="-1"><a class="header-anchor" href="#step-5-evaluate-the-impact-of-missing-value-handling-on-subsequent-analysis"><span>Step 5: Evaluate the Impact of Missing Value Handling on Subsequent Analysis</span></a></h3>
<p>Finally, we evaluate the impact of the missing value handling on a disease prediction model.</p>
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token comment"># Prepare features and target variable for prediction</span></span>
<span class="line">X <span class="token operator">=</span> df_complete<span class="token punctuation">.</span>drop<span class="token punctuation">(</span><span class="token punctuation">[</span><span class="token string">'patient_id'</span><span class="token punctuation">,</span> <span class="token string">'diagnosis'</span><span class="token punctuation">]</span> <span class="token operator">+</span> </span>
<span class="line">                    <span class="token punctuation">[</span>c <span class="token keyword">for</span> c <span class="token keyword">in</span> df_complete<span class="token punctuation">.</span>columns <span class="token keyword">if</span> c<span class="token punctuation">.</span>endswith<span class="token punctuation">(</span><span class="token string">'_missing'</span><span class="token punctuation">)</span><span class="token punctuation">]</span><span class="token punctuation">,</span> axis<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span></span>
<span class="line">y <span class="token operator">=</span> df_complete<span class="token punctuation">[</span><span class="token string">'diagnosis'</span><span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Convert categorical features into numerical values</span></span>
<span class="line">X <span class="token operator">=</span> pd<span class="token punctuation">.</span>get_dummies<span class="token punctuation">(</span>X<span class="token punctuation">,</span> drop_first<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Split into training and test sets</span></span>
<span class="line">X_train<span class="token punctuation">,</span> X_test<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> y_test <span class="token operator">=</span> train_test_split<span class="token punctuation">(</span>X<span class="token punctuation">,</span> y<span class="token punctuation">,</span> test_size<span class="token operator">=</span><span class="token number">0.2</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Train a RandomForest classifier</span></span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>ensemble <span class="token keyword">import</span> RandomForestClassifier</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>metrics <span class="token keyword">import</span> classification_report<span class="token punctuation">,</span> confusion_matrix</span>
<span class="line"></span>
<span class="line">clf <span class="token operator">=</span> RandomForestClassifier<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">clf<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train<span class="token punctuation">,</span> y_train<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Predict and evaluate</span></span>
<span class="line">y_pred <span class="token operator">=</span> clf<span class="token punctuation">.</span>predict<span class="token punctuation">(</span>X_test<span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Classification Report:"</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>classification_report<span class="token punctuation">(</span>y_test<span class="token punctuation">,</span> y_pred<span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Visualize the confusion matrix</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">8</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>heatmap<span class="token punctuation">(</span>confusion_matrix<span class="token punctuation">(</span>y_test<span class="token punctuation">,</span> y_pred<span class="token punctuation">)</span><span class="token punctuation">,</span> annot<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span> fmt<span class="token operator">=</span><span class="token string">'d'</span><span class="token punctuation">,</span> cmap<span class="token operator">=</span><span class="token string">'Blues'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'Predicted Label'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'True Label'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Confusion Matrix'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Feature importance</span></span>
<span class="line">feature_importance <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span><span class="token punctuation">{</span></span>
<span class="line">    <span class="token string">'feature'</span><span class="token punctuation">:</span> X<span class="token punctuation">.</span>columns<span class="token punctuation">,</span></span>
<span class="line">    <span class="token string">'importance'</span><span class="token punctuation">:</span> clf<span class="token punctuation">.</span>feature_importances_</span>
<span class="line"><span class="token punctuation">}</span><span class="token punctuation">)</span><span class="token punctuation">.</span>sort_values<span class="token punctuation">(</span><span class="token string">'importance'</span><span class="token punctuation">,</span> ascending<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>barplot<span class="token punctuation">(</span>x<span class="token operator">=</span><span class="token string">'importance'</span><span class="token punctuation">,</span> y<span class="token operator">=</span><span class="token string">'feature'</span><span class="token punctuation">,</span> data<span class="token operator">=</span>feature_importance<span class="token punctuation">.</span>head<span class="token punctuation">(</span><span class="token number">15</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Feature Importance'</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>tight_layout<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="results-analysis" tabindex="-1"><a class="header-anchor" href="#results-analysis"><span>Results Analysis</span></a></h2>
<p>By comparing different missing value handling methods, we reached the following conclusions:</p>
<ol>
<li><strong>Iterative Imputation</strong> performed best for most features, especially those with high correlation.</li>
<li><strong>KNN Imputation</strong> performed well on some features but was computationally expensive.</li>
<li><strong>Group-based Imputation</strong> was effective for features strongly related to grouping variables.</li>
<li><strong>Simple Imputation</strong> is straightforward but generally less accurate.</li>
</ol>
<p>The chosen iterative imputation method successfully handled the missing values in the dataset, providing high-quality input data for subsequent disease prediction models. The prediction model achieved good performance on the test set, indicating that our missing value handling strategy was effective.</p>
<h2 id="advanced-challenges" tabindex="-1"><a class="header-anchor" href="#advanced-challenges"><span>Advanced Challenges</span></a></h2>
<p>If you have completed the basic tasks, you can try the following advanced challenges:</p>
<ol>
<li><strong>Analysis of Missing Mechanisms</strong>: Delve into the missing mechanisms (MCAR, MAR, MNAR).</li>
<li><strong>Sensitivity Analysis</strong>: Evaluate how sensitive the final model results are to different imputation methods.</li>
<li><strong>Advanced Multiple Imputation</strong>: Implement a complete multiple imputation process, including generating multiple imputed datasets and combining the results.</li>
<li><strong>Custom Imputation Models</strong>: Develop tailored predictive models for imputing specific features.</li>
<li><strong>Missing Value Simulation</strong>: Design experiments by simulating different missing patterns on complete data to assess the robustness of various methods.</li>
</ol>
<h2 id="summary-and-reflection" tabindex="-1"><a class="header-anchor" href="#summary-and-reflection"><span>Summary and Reflection</span></a></h2>
<p>Through this project, we learned how to handle missing values in medical data and compare the effectiveness of different methods. Missing value handling is a crucial step in medical data analysis as it directly affects the accuracy of subsequent analyses and predictions.</p>
<p>In practical applications, these techniques can help healthcare institutions better utilize incomplete patient data, thereby improving the accuracy of disease prediction and diagnosis. For example, with appropriate handling, even when some test results are missing, relatively accurate risk assessments can be provided to patients.</p>
<h3 id="reflection-questions" tabindex="-1"><a class="header-anchor" href="#reflection-questions"><span>Reflection Questions</span></a></h3>
<ol>
<li>In medical data, missing values may carry information (e.g., the absence of a test might indicate that a doctor did not deem it necessary). How can we retain this information while imputing the missing values?</li>
<li>Different types of medical data (such as lab tests, surveys, imaging data) may require different imputation strategies. How can we choose the appropriate method for different types of data?</li>
<li>When handling sensitive medical data, how can we balance data completeness with the need for privacy protection?</li>
</ol>
<div class="practice-link">
  <a href="/en/projects/classification/titanic.html" class="button">Next Module: Classification Project</a>
</div>
</div></template>


