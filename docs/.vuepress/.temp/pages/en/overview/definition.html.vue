<template><div><h1 id="definition-and-basic-concepts-of-data-mining" tabindex="-1"><a class="header-anchor" href="#definition-and-basic-concepts-of-data-mining"><span>Definition and Basic Concepts of Data Mining</span></a></h1>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">📚</span> Key Points of This Section
  </div>
  <div class="knowledge-card__content">
    <ul>
      <li>Understand the definition and core objectives of data mining</li>
      <li>Master the relationship between data mining and related disciplines</li>
      <li>Understand the main tasks in data mining</li>
      <li>Recognize the importance of data mining in modern society</li>
    </ul>
  </div>
</div>
<h2 id="what-is-data-mining" tabindex="-1"><a class="header-anchor" href="#what-is-data-mining"><span>What is Data Mining?</span></a></h2>
<p>Data mining is the process of extracting valuable patterns, relationships, and knowledge from large datasets. It combines methods from statistics, machine learning, database technology, and artificial intelligence, aiming to uncover meaningful information from seemingly chaotic data.</p>
<h3 id="formal-definition-of-data-mining" tabindex="-1"><a class="header-anchor" href="#formal-definition-of-data-mining"><span>Formal Definition of Data Mining</span></a></h3>
<p>The definition provided by the American Computer Association (ACM):</p>
<blockquote>
<p>Data mining is the process of extracting previously unknown, potentially useful information from large datasets.</p>
</blockquote>
<p>This definition emphasizes several key features of data mining:</p>
<ul>
<li>Handling <strong>large datasets</strong></li>
<li>Discovering <strong>unknown information</strong></li>
<li>Extracting <strong>useful knowledge</strong></li>
<li>A <strong>systematic process</strong></li>
</ul>
<h2 id="relationship-between-data-mining-and-related-disciplines" tabindex="-1"><a class="header-anchor" href="#relationship-between-data-mining-and-related-disciplines"><span>Relationship Between Data Mining and Related Disciplines</span></a></h2>
<p>Data mining is an interdisciplinary field closely related to several other areas:</p>
<DisciplineMap /><h3 id="difference-between-data-mining-and-machine-learning" tabindex="-1"><a class="header-anchor" href="#difference-between-data-mining-and-machine-learning"><span>Difference Between Data Mining and Machine Learning</span></a></h3>
<p>Although data mining and machine learning are closely related, they have important differences:</p>
<ul>
<li><strong>Scope</strong>: Data mining is a broader process that includes data preparation, pattern discovery, and result interpretation; whereas machine learning focuses mainly on the development of algorithms and models.</li>
<li><strong>Goal</strong>: Data mining focuses on discovering useful knowledge and insights; machine learning focuses more on prediction and automated decision-making.</li>
<li><strong>Methods</strong>: Data mining uses a variety of techniques, including but not limited to machine learning; machine learning is an important tool within data mining.</li>
</ul>
<h2 id="main-tasks-of-data-mining" tabindex="-1"><a class="header-anchor" href="#main-tasks-of-data-mining"><span>Main Tasks of Data Mining</span></a></h2>
<p>Data mining can address various types of problems, mainly including:</p>
<h3 id="_1-descriptive-tasks" tabindex="-1"><a class="header-anchor" href="#_1-descriptive-tasks"><span>1. Descriptive Tasks</span></a></h3>
<p>Descriptive tasks aim to understand and describe the inherent characteristics and patterns in the data.</p>
<ul>
<li><strong>Clustering Analysis</strong>: Grouping similar data points to discover natural classifications</li>
<li><strong>Association Rule Mining</strong>: Discovering relationships between items in the data, such as &quot;customers who buy diapers are also likely to buy beer&quot;</li>
<li><strong>Anomaly Detection</strong>: Identifying data points that significantly differ from normal patterns</li>
<li><strong>Summarization</strong>: Generating concise representations or summaries of the data</li>
</ul>
<h3 id="_2-predictive-tasks" tabindex="-1"><a class="header-anchor" href="#_2-predictive-tasks"><span>2. Predictive Tasks</span></a></h3>
<p>Predictive tasks aim to forecast future or unknown values based on historical data.</p>
<ul>
<li><strong>Classification</strong>: Predicting discrete category labels, such as spam detection</li>
<li><strong>Regression</strong>: Predicting continuous values, such as house price prediction</li>
<li><strong>Time Series Analysis</strong>: Predicting data that changes over time, such as stock price prediction</li>
<li><strong>Recommendation Systems</strong>: Predicting user preferences and recommending relevant items</li>
</ul>
<div class="code-example">
  <div class="code-example__title">Task Example: Classification</div>
  <div class="code-example__content">
<div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre v-pre><code><span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>ensemble <span class="token keyword">import</span> RandomForestClassifier</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>model_selection <span class="token keyword">import</span> train_test_split</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>metrics <span class="token keyword">import</span> accuracy_score</span>
<span class="line"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd</span>
<span class="line"></span>
<span class="line"><span class="token comment"># Load data</span></span>
<span class="line">data <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'customer_churn.csv'</span><span class="token punctuation">)</span></span>
<span class="line">X <span class="token operator">=</span> data<span class="token punctuation">.</span>drop<span class="token punctuation">(</span><span class="token string">'Churn'</span><span class="token punctuation">,</span> axis<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">)</span></span>
<span class="line">y <span class="token operator">=</span> data<span class="token punctuation">[</span><span class="token string">'Churn'</span><span class="token punctuation">]</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Split into training and test sets</span></span>
<span class="line">X_train<span class="token punctuation">,</span> X_test<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> y_test <span class="token operator">=</span> train_test_split<span class="token punctuation">(</span>X<span class="token punctuation">,</span> y<span class="token punctuation">,</span> test_size<span class="token operator">=</span><span class="token number">0.3</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Train classification model</span></span>
<span class="line">model <span class="token operator">=</span> RandomForestClassifier<span class="token punctuation">(</span>n_estimators<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">model<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train<span class="token punctuation">,</span> y_train<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Predictions and evaluation</span></span>
<span class="line">y_pred <span class="token operator">=</span> model<span class="token punctuation">.</span>predict<span class="token punctuation">(</span>X_test<span class="token punctuation">)</span></span>
<span class="line">accuracy <span class="token operator">=</span> accuracy_score<span class="token punctuation">(</span>y_test<span class="token punctuation">,</span> y_pred<span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"Model Accuracy: </span><span class="token interpolation"><span class="token punctuation">{</span>accuracy<span class="token punctuation">:</span><span class="token format-spec">.4f</span><span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre>
<div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>  </div>
</div>
<h2 id="importance-of-data-mining" tabindex="-1"><a class="header-anchor" href="#importance-of-data-mining"><span>Importance of Data Mining</span></a></h2>
<p>In today's digital age, data mining has become increasingly important for several reasons:</p>
<ol>
<li><strong>Data Explosion</strong>: The global volume of data is growing exponentially, requiring automated tools to extract value.</li>
<li><strong>Business Competition</strong>: Data-driven decision-making has become a key source of competitive advantage.</li>
<li><strong>Scientific Discovery</strong>: Data mining accelerates scientific research and discoveries.</li>
<li><strong>Personalized Services</strong>: Enables businesses to offer customized products and services.</li>
<li><strong>Resource Optimization</strong>: Helps organizations optimize resource allocation and operational efficiency.</li>
</ol>
<div class="knowledge-card">
  <div class="knowledge-card__title">
    <span class="icon">💡</span> Did You Know?
  </div>
  <div class="knowledge-card__content">
    <p>The term "data mining" first appeared in the early 1990s, but its core concepts can be traced back to earlier research in statistical analysis and pattern recognition. With the advancement of computational power and the growth of data volume, data mining has evolved from an academic concept to an important business and research tool.</p>
  </div>
</div>
<h2 id="summary-and-reflection" tabindex="-1"><a class="header-anchor" href="#summary-and-reflection"><span>Summary and Reflection</span></a></h2>
<p>Data mining is the process of extracting valuable knowledge from large datasets, combining methods and techniques from multiple disciplines. It can address a wide range of descriptive and predictive tasks.</p>
<h3 id="key-takeaways" tabindex="-1"><a class="header-anchor" href="#key-takeaways"><span>Key Takeaways</span></a></h3>
<ul>
<li>Data mining is the process of extracting valuable patterns from large datasets</li>
<li>It is closely related to fields such as statistics, machine learning, and database technologies</li>
<li>Major tasks include clustering, classification, regression, and association rule mining</li>
<li>The importance of data mining is increasingly evident in today's data-driven world</li>
</ul>
<h3 id="reflective-questions" tabindex="-1"><a class="header-anchor" href="#reflective-questions"><span>Reflective Questions</span></a></h3>
<ol>
<li>How does data mining change decision-making in your industry?</li>
<li>Can you think of any real-world examples of data mining applications in daily life?</li>
<li>What technical and ethical challenges does data mining face?</li>
</ol>
<BackToPath /><div class="practice-link">
  <a href="/en/overview/process.html" class="button">Next Section: Data Mining Process</a>
</div>
</div></template>


