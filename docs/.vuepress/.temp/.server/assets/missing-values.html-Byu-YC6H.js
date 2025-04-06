import { resolveComponent, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderStyle, ssrRenderAttr, ssrRenderComponent } from "vue/server-renderer";
import { _ as _imports_0 } from "./missing_pattern_en-C7yjWJ-7.js";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  const _component_missing_value_imputation = resolveComponent("missing-value-imputation");
  const _component_BackToPath = resolveComponent("BackToPath");
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="missing-value-handling" tabindex="-1"><a class="header-anchor" href="#missing-value-handling"><span>Missing Value Handling</span></a></h1><div class="knowledge-card"><div class="knowledge-card__title"><span class="icon">📚</span>Key Points of This Section </div><div class="knowledge-card__content"><ul><li>Understand the causes and impacts of missing values</li><li>Master methods for detecting and analyzing missing values</li><li>Learn various strategies for handling missing values and their appropriate scenarios</li><li>Practice common techniques for missing value processing</li></ul></div></div><h2 id="overview-of-missing-values" tabindex="-1"><a class="header-anchor" href="#overview-of-missing-values"><span>Overview of Missing Values</span></a></h2><p>Missing values are a common problem in data analysis and machine learning. Effectively handling missing values is crucial for model performance.</p><h3 id="causes-of-missing-values" tabindex="-1"><a class="header-anchor" href="#causes-of-missing-values"><span>Causes of Missing Values</span></a></h3><p>Missing values are usually caused by the following reasons:</p><ol><li><strong>Data Collection Issues</strong>: For example, unanswered survey questions or sensor failures.</li><li><strong>Data Integration Issues</strong>: Incomplete information when merging data from multiple sources.</li><li><strong>Data Processing Errors</strong>: Errors during import, transformation, or cleaning processes.</li><li><strong>Privacy Protection</strong>: Intentionally hiding sensitive information.</li><li><strong>Structural Missingness</strong>: Data not required under certain conditions.</li></ol><h3 id="types-of-missing-values" tabindex="-1"><a class="header-anchor" href="#types-of-missing-values"><span>Types of Missing Values</span></a></h3><p>Based on the missing mechanism, missing values can be classified into three types:</p><ol><li><strong>Missing Completely At Random (MCAR)</strong><ul><li>The missingness occurs completely at random and is unrelated to any observed or unobserved variables.</li><li>For example: Data loss due to random equipment failure.</li></ul></li><li><strong>Missing At Random (MAR)</strong><ul><li>The probability of missing depends solely on other observed variables.</li><li>For example: Older individuals might be less likely to disclose their income.</li></ul></li><li><strong>Missing Not At Random (MNAR)</strong><ul><li>The probability of missing is related to unobserved variables or the missing value itself.</li><li>For example: High-income individuals might choose not to reveal their income.</li></ul></li></ol><div class="knowledge-card"><div class="knowledge-card__title"><span class="icon">💡</span>Did You Know? </div><div class="knowledge-card__content"><p>Statisticians Donald Rubin first systematically proposed the classification of missing data mechanisms (MCAR, MAR, MNAR) in 1976. This framework remains the foundation for handling missing values today. Different missing mechanisms require different handling strategies, and selecting the right method is crucial for reliable analysis.</p></div></div><h2 id="analyzing-missing-values" tabindex="-1"><a class="header-anchor" href="#analyzing-missing-values"><span>Analyzing Missing Values</span></a></h2><h3 id="_1-detecting-missing-values" tabindex="-1"><a class="header-anchor" href="#_1-detecting-missing-values"><span>1. Detecting Missing Values</span></a></h3><p>Before handling missing values, it is essential to thoroughly understand their extent in the data:</p><div class="code-example"><div class="code-example__title">Code Example</div><div class="code-example__content"><div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre><code><span class="line"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd</span>
<span class="line"><span class="token keyword">import</span> numpy <span class="token keyword">as</span> np</span>
<span class="line"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt</span>
<span class="line"><span class="token keyword">import</span> seaborn <span class="token keyword">as</span> sns</span>
<span class="line"></span>
<span class="line"><span class="token comment"># Load data</span></span>
<span class="line">df <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">&#39;data.csv&#39;</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Check the number of missing values per column</span></span>
<span class="line">missing_values <span class="token operator">=</span> df<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>missing_values<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Calculate the missing ratio</span></span>
<span class="line">missing_ratio <span class="token operator">=</span> df<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span><span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">/</span> <span class="token builtin">len</span><span class="token punctuation">(</span>df<span class="token punctuation">)</span> <span class="token operator">*</span> <span class="token number">100</span></span>
<span class="line"><span class="token keyword">print</span><span class="token punctuation">(</span>missing_ratio<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Visualize missing values</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">6</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">missing_ratio <span class="token operator">=</span> missing_ratio<span class="token punctuation">[</span>missing_ratio <span class="token operator">&gt;</span> <span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">.</span>sort_values<span class="token punctuation">(</span>ascending<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>barplot<span class="token punctuation">(</span>x<span class="token operator">=</span>missing_ratio<span class="token punctuation">.</span>index<span class="token punctuation">,</span> y<span class="token operator">=</span>missing_ratio<span class="token punctuation">.</span>values<span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">&#39;Missing Value Ratio&#39;</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>xticks<span class="token punctuation">(</span>rotation<span class="token operator">=</span><span class="token number">45</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">&#39;Percentage Missing&#39;</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="${ssrRenderStyle({ "counter-reset": "line-number 0" })}"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div></div></div><h3 id="_2-analyzing-missing-patterns" tabindex="-1"><a class="header-anchor" href="#_2-analyzing-missing-patterns"><span>2. Analyzing Missing Patterns</span></a></h3><p>Understanding the relationships among missing values helps in choosing the right handling strategy:</p><div class="code-example"><div class="code-example__title">Code Example</div><div class="code-example__content"><div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre><code><span class="line"><span class="token comment"># Missing value heatmap</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">12</span><span class="token punctuation">,</span> <span class="token number">8</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>heatmap<span class="token punctuation">(</span>df<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> cbar<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">,</span> cmap<span class="token operator">=</span><span class="token string">&#39;viridis&#39;</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">&#39;Missing Value Distribution Heatmap&#39;</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Correlation of missing values</span></span>
<span class="line">missing_binary <span class="token operator">=</span> df<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>astype<span class="token punctuation">(</span><span class="token builtin">int</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">8</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>heatmap<span class="token punctuation">(</span>missing_binary<span class="token punctuation">.</span>corr<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> annot<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span> cmap<span class="token operator">=</span><span class="token string">&#39;coolwarm&#39;</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">&#39;Missing Value Correlation Heatmap&#39;</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="${ssrRenderStyle({ "counter-reset": "line-number 0" })}"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div></div></div><div class="visualization-container"><div class="visualization-title">Visualization of Missing Patterns</div><div class="visualization-content"><img${ssrRenderAttr("src", _imports_0)} alt="Visualization of Missing Patterns"></div><div class="visualization-caption"> Figure: Heatmap showing the distribution of missing values. The yellow areas indicate missing values, allowing observation of missing patterns. </div></div><h2 id="strategies-for-handling-missing-values" tabindex="-1"><a class="header-anchor" href="#strategies-for-handling-missing-values"><span>Strategies for Handling Missing Values</span></a></h2><h3 id="_1-deleting-data-with-missing-values" tabindex="-1"><a class="header-anchor" href="#_1-deleting-data-with-missing-values"><span>1. Deleting Data with Missing Values</span></a></h3><p>The simplest method is to delete rows or columns with missing values, though it may lead to information loss:</p><div class="code-example"><div class="code-example__title">Code Example</div><div class="code-example__content"><div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre><code><span class="line"><span class="token comment"># Delete all rows with missing values</span></span>
<span class="line">df_dropped_rows <span class="token operator">=</span> df<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Delete columns with more than 50% missing values</span></span>
<span class="line">threshold <span class="token operator">=</span> <span class="token builtin">len</span><span class="token punctuation">(</span>df<span class="token punctuation">)</span> <span class="token operator">*</span> <span class="token number">0.5</span></span>
<span class="line">df_dropped_cols <span class="token operator">=</span> df<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span>axis<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">,</span> thresh<span class="token operator">=</span>threshold<span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Delete rows with missing values in specific columns</span></span>
<span class="line">df_dropped_specific <span class="token operator">=</span> df<span class="token punctuation">.</span>dropna<span class="token punctuation">(</span>subset<span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">&#39;income&#39;</span><span class="token punctuation">,</span> <span class="token string">&#39;age&#39;</span><span class="token punctuation">]</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="${ssrRenderStyle({ "counter-reset": "line-number 0" })}"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div></div></div><p><strong>Applicable Scenarios</strong>:</p><ul><li>When the missing ratio is very low (less than 5%).</li><li>When the dataset is large enough so that deletion doesn&#39;t significantly reduce the sample size.</li><li>When missing values occur completely at random (MCAR).</li></ul><p><strong>Drawbacks</strong>:</p><ul><li>May introduce bias, especially if missingness is not completely random.</li><li>Reduces the overall sample size, potentially lowering statistical power.</li><li>Might result in the loss of valuable information.</li></ul><h3 id="_2-imputing-missing-values" tabindex="-1"><a class="header-anchor" href="#_2-imputing-missing-values"><span>2. Imputing Missing Values</span></a></h3><h4 id="_2-1-imputation-using-statistical-measures" tabindex="-1"><a class="header-anchor" href="#_2-1-imputation-using-statistical-measures"><span>2.1 Imputation Using Statistical Measures</span></a></h4><p>Replace missing values with statistics such as mean, median, or mode:</p><div class="code-example"><div class="code-example__title">Code Example</div><div class="code-example__content"><div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre><code><span class="line"><span class="token comment"># Mean imputation</span></span>
<span class="line">df_mean <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">df_mean<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_mean<span class="token punctuation">.</span>mean<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> inplace<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Median imputation</span></span>
<span class="line">df_median <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">df_median<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_median<span class="token punctuation">.</span>median<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> inplace<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Mode imputation for categorical variables</span></span>
<span class="line">df_mode <span class="token operator">=</span> df<span class="token punctuation">.</span>copy<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token keyword">for</span> col <span class="token keyword">in</span> df_mode<span class="token punctuation">.</span>columns<span class="token punctuation">:</span></span>
<span class="line">    <span class="token keyword">if</span> df_mode<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>dtype <span class="token operator">==</span> <span class="token string">&#39;object&#39;</span><span class="token punctuation">:</span></span>
<span class="line">        df_mode<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>fillna<span class="token punctuation">(</span>df_mode<span class="token punctuation">[</span>col<span class="token punctuation">]</span><span class="token punctuation">.</span>mode<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span><span class="token punctuation">,</span> inplace<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="${ssrRenderStyle({ "counter-reset": "line-number 0" })}"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div></div></div><div class="interactive-component"><div class="interactive-title">Comparison of Imputation Methods</div><div class="interactive-content">`);
  _push(ssrRenderComponent(_component_missing_value_imputation, null, null, _parent));
  _push(`</div><div class="interactive-caption"> Interactive Component: Experiment with different imputation methods and observe their effects on the data distribution. </div></div><h4 id="_2-2-advanced-imputation-methods" tabindex="-1"><a class="header-anchor" href="#_2-2-advanced-imputation-methods"><span>2.2 Advanced Imputation Methods</span></a></h4><p>Use data relationships to perform more sophisticated imputations:</p><div class="code-example"><div class="code-example__title">Code Example</div><div class="code-example__content"><div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre><code><span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>impute <span class="token keyword">import</span> KNNImputer</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>experimental <span class="token keyword">import</span> enable_iterative_imputer</span>
<span class="line"><span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>impute <span class="token keyword">import</span> IterativeImputer</span>
<span class="line"></span>
<span class="line"><span class="token comment"># KNN imputation</span></span>
<span class="line">imputer <span class="token operator">=</span> KNNImputer<span class="token punctuation">(</span>n_neighbors<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">)</span></span>
<span class="line">df_knn <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span></span>
<span class="line">    imputer<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df<span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    columns<span class="token operator">=</span>df<span class="token punctuation">.</span>columns</span>
<span class="line"><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Multiple Imputation using a simplified version of the MICE algorithm</span></span>
<span class="line">imputer <span class="token operator">=</span> IterativeImputer<span class="token punctuation">(</span>max_iter<span class="token operator">=</span><span class="token number">10</span><span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span></span>
<span class="line">df_mice <span class="token operator">=</span> pd<span class="token punctuation">.</span>DataFrame<span class="token punctuation">(</span></span>
<span class="line">    imputer<span class="token punctuation">.</span>fit_transform<span class="token punctuation">(</span>df<span class="token punctuation">)</span><span class="token punctuation">,</span></span>
<span class="line">    columns<span class="token operator">=</span>df<span class="token punctuation">.</span>columns</span>
<span class="line"><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="${ssrRenderStyle({ "counter-reset": "line-number 0" })}"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div></div></div><h2 id="evaluating-missing-value-handling" tabindex="-1"><a class="header-anchor" href="#evaluating-missing-value-handling"><span>Evaluating Missing Value Handling</span></a></h2><h3 id="_1-comparing-different-methods" tabindex="-1"><a class="header-anchor" href="#_1-comparing-different-methods"><span>1. Comparing Different Methods</span></a></h3><div class="code-example"><div class="code-example__title">Code Example</div><div class="code-example__content"><div class="language-python line-numbers-mode" data-highlighter="prismjs" data-ext="py"><pre><code><span class="line"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt</span>
<span class="line"><span class="token keyword">import</span> seaborn <span class="token keyword">as</span> sns</span>
<span class="line"></span>
<span class="line"><span class="token comment"># Original data distribution</span></span>
<span class="line">plt<span class="token punctuation">.</span>figure<span class="token punctuation">(</span>figsize<span class="token operator">=</span><span class="token punctuation">(</span><span class="token number">15</span><span class="token punctuation">,</span> <span class="token number">5</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>subplot<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>histplot<span class="token punctuation">(</span>df<span class="token punctuation">[</span><span class="token string">&#39;income&#39;</span><span class="token punctuation">]</span><span class="token punctuation">.</span>dropna<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> kde<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">&#39;Original Data Distribution&#39;</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Distribution after mean imputation</span></span>
<span class="line">plt<span class="token punctuation">.</span>subplot<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>histplot<span class="token punctuation">(</span>df_mean<span class="token punctuation">[</span><span class="token string">&#39;income&#39;</span><span class="token punctuation">]</span><span class="token punctuation">,</span> kde<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">&#39;Distribution After Mean Imputation&#39;</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line"><span class="token comment"># Distribution after KNN imputation</span></span>
<span class="line">plt<span class="token punctuation">.</span>subplot<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">)</span></span>
<span class="line">sns<span class="token punctuation">.</span>histplot<span class="token punctuation">(</span>df_knn<span class="token punctuation">[</span><span class="token string">&#39;income&#39;</span><span class="token punctuation">]</span><span class="token punctuation">,</span> kde<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">&#39;Distribution After KNN Imputation&#39;</span><span class="token punctuation">)</span></span>
<span class="line"></span>
<span class="line">plt<span class="token punctuation">.</span>tight_layout<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line">plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="${ssrRenderStyle({ "counter-reset": "line-number 0" })}"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div></div></div><h2 id="practical-recommendations" tabindex="-1"><a class="header-anchor" href="#practical-recommendations"><span>Practical Recommendations</span></a></h2><h3 id="_1-missing-value-handling-workflow" tabindex="-1"><a class="header-anchor" href="#_1-missing-value-handling-workflow"><span>1. Missing Value Handling Workflow</span></a></h3><ol><li><strong>Exploratory Analysis</strong>: <ul><li>Detect the locations, counts, and ratios of missing values.</li><li>Analyze missing patterns and mechanisms.</li><li>Visualize the distribution of missing values.</li></ul></li><li><strong>Develop a Handling Strategy</strong>: <ul><li>Choose an appropriate method based on the missing mechanism.</li><li>Consider feature importance and data structure.</li><li>Use different strategies for different features if needed.</li></ul></li><li><strong>Implementation and Evaluation</strong>: <ul><li>Apply the selected imputation method.</li><li>Compare the effects of various methods.</li><li>Validate the quality and consistency of the processed data.</li></ul></li></ol><h3 id="_2-practical-tips" tabindex="-1"><a class="header-anchor" href="#_2-practical-tips"><span>2. Practical Tips</span></a></h3><ul><li><strong>Analyze Before Processing</strong>: Understand the underlying reasons for missing values before choosing a method.</li><li><strong>Consider Feature Correlations</strong>: Utilize the relationships between features to enhance imputation.</li><li><strong>Incorporate Domain Knowledge</strong>: Let business or domain insights guide your missing value handling.</li><li><strong>Perform Sensitivity Analysis</strong>: Test how different imputation methods affect your final results.</li><li><strong>Preserve Uncertainty</strong>: Consider methods like multiple imputation to capture the uncertainty in estimates.</li></ul><div class="knowledge-card"><div class="knowledge-card__title"><span class="icon">⚠️</span>Common Misconceptions </div><div class="knowledge-card__content"><ul><li><strong>Blind Deletion</strong>: Removing samples with missing values without analyzing the underlying mechanisms.</li><li><strong>Overreliance on Mean Imputation</strong>: Using the mean for all features without considering their distribution.</li><li><strong>Ignoring Missing Correlations</strong>: Failing to consider relationships between features during imputation.</li><li><strong>Data Leakage</strong>: Using information from the test set to impute the training set.</li></ul></div></div><h2 id="summary-and-reflection" tabindex="-1"><a class="header-anchor" href="#summary-and-reflection"><span>Summary and Reflection</span></a></h2><p>Handling missing values is a critical step in data preprocessing. The right method can enhance model performance and ensure reliable analysis results.</p><h3 id="key-takeaways" tabindex="-1"><a class="header-anchor" href="#key-takeaways"><span>Key Takeaways</span></a></h3><ul><li>Missing values can arise from various causes, including data collection issues, privacy protection, and more.</li><li>Missing mechanisms are classified into MCAR, MAR, and MNAR.</li><li>Handling strategies include both deletion and various imputation methods.</li><li>The choice of method should consider the missing mechanism, data characteristics, and analysis objectives.</li></ul><h3 id="reflection-questions" tabindex="-1"><a class="header-anchor" href="#reflection-questions"><span>Reflection Questions</span></a></h3><ol><li>How can we determine the mechanism behind missing values in a dataset?</li><li>Under what circumstances is it more appropriate to delete samples with missing values rather than impute them?</li><li>How can we evaluate the effectiveness of different missing value handling methods?</li></ol>`);
  _push(ssrRenderComponent(_component_BackToPath, null, null, _parent));
  _push(`<div class="practice-link"><a href="/projects/preprocessing.html" class="button">Proceed to Practice Projects</a></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/en/core/preprocessing/missing-values.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const missingValues_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "missing-values.html.vue"]]);
const data = JSON.parse('{"path":"/en/core/preprocessing/missing-values.html","title":"Missing Value Handling","lang":"en-US","frontmatter":{},"headers":[{"level":2,"title":"Overview of Missing Values","slug":"overview-of-missing-values","link":"#overview-of-missing-values","children":[{"level":3,"title":"Causes of Missing Values","slug":"causes-of-missing-values","link":"#causes-of-missing-values","children":[]},{"level":3,"title":"Types of Missing Values","slug":"types-of-missing-values","link":"#types-of-missing-values","children":[]}]},{"level":2,"title":"Analyzing Missing Values","slug":"analyzing-missing-values","link":"#analyzing-missing-values","children":[{"level":3,"title":"1. Detecting Missing Values","slug":"_1-detecting-missing-values","link":"#_1-detecting-missing-values","children":[]},{"level":3,"title":"2. Analyzing Missing Patterns","slug":"_2-analyzing-missing-patterns","link":"#_2-analyzing-missing-patterns","children":[]}]},{"level":2,"title":"Strategies for Handling Missing Values","slug":"strategies-for-handling-missing-values","link":"#strategies-for-handling-missing-values","children":[{"level":3,"title":"1. Deleting Data with Missing Values","slug":"_1-deleting-data-with-missing-values","link":"#_1-deleting-data-with-missing-values","children":[]},{"level":3,"title":"2. Imputing Missing Values","slug":"_2-imputing-missing-values","link":"#_2-imputing-missing-values","children":[]}]},{"level":2,"title":"Evaluating Missing Value Handling","slug":"evaluating-missing-value-handling","link":"#evaluating-missing-value-handling","children":[{"level":3,"title":"1. Comparing Different Methods","slug":"_1-comparing-different-methods","link":"#_1-comparing-different-methods","children":[]}]},{"level":2,"title":"Practical Recommendations","slug":"practical-recommendations","link":"#practical-recommendations","children":[{"level":3,"title":"1. Missing Value Handling Workflow","slug":"_1-missing-value-handling-workflow","link":"#_1-missing-value-handling-workflow","children":[]},{"level":3,"title":"2. Practical Tips","slug":"_2-practical-tips","link":"#_2-practical-tips","children":[]}]},{"level":2,"title":"Summary and Reflection","slug":"summary-and-reflection","link":"#summary-and-reflection","children":[{"level":3,"title":"Key Takeaways","slug":"key-takeaways","link":"#key-takeaways","children":[]},{"level":3,"title":"Reflection Questions","slug":"reflection-questions","link":"#reflection-questions","children":[]}]}],"git":{"updatedTime":1742831857000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":2,"url":"https://github.com/like45599"}],"changelog":[{"hash":"2bc457cfaf02a69e1673760e9106a75f7cced3da","time":1742831857000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"优化跳转地址+更新网站icon"},{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"en/core/preprocessing/missing-values.md"}');
export {
  missingValues_html as comp,
  data
};
