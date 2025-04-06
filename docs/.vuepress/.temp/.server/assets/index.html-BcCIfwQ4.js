import { resolveComponent, withCtx, createVNode, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderStyle, ssrRenderComponent } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  const _component_ClientOnly = resolveComponent("ClientOnly");
  const _component_practice_notebook = resolveComponent("practice-notebook");
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="数据挖掘实践指南" tabindex="-1"><a class="header-anchor" href="#数据挖掘实践指南"><span>数据挖掘实践指南</span></a></h1><h2 id="在线实践环境" tabindex="-1"><a class="header-anchor" href="#在线实践环境"><span>在线实践环境</span></a></h2><div class="hint-container tip"><p class="hint-container-title">推荐使用</p><p>我们推荐以下在线环境进行代码实践：</p><ol><li><a href="https://www.aliyun.com/product/bigdata/dsw" target="_blank" rel="noopener noreferrer">阿里云 DSW</a> - 数据科学工作环境</li><li><a href="https://aistudio.baidu.com/" target="_blank" rel="noopener noreferrer">百度 AI Studio</a> - 免费的深度学习平台</li><li><a href="https://www.heywhale.com/" target="_blank" rel="noopener noreferrer">和鲸社区</a> - 国内数据科学竞赛平台</li></ol></div><h2 id="本地环境配置" tabindex="-1"><a class="header-anchor" href="#本地环境配置"><span>本地环境配置</span></a></h2><div class="hint-container warning"><p class="hint-container-title">环境配置</p><p>推荐使用 Anaconda 配置本地环境：</p><ol><li>下载 <a href="https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/" target="_blank" rel="noopener noreferrer">Anaconda</a></li><li>使用清华源加速：</li></ol><div class="language-bash line-numbers-mode" data-highlighter="prismjs" data-ext="sh"><pre><code><span class="line">conda config <span class="token parameter variable">--add</span> channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/</span>
<span class="line">conda config <span class="token parameter variable">--add</span> channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/</span>
<span class="line">conda config <span class="token parameter variable">--set</span> show_channel_urls <span class="token function">yes</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="${ssrRenderStyle({ "counter-reset": "line-number 0" })}"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><ol start="3"><li>创建项目环境：</li></ol><div class="language-bash line-numbers-mode" data-highlighter="prismjs" data-ext="sh"><pre><code><span class="line">conda create <span class="token parameter variable">-n</span> datamining <span class="token assign-left variable">python</span><span class="token operator">=</span><span class="token number">3.8</span></span>
<span class="line">conda activate datamining</span>
<span class="line">pip <span class="token function">install</span> <span class="token parameter variable">-r</span> requirements.txt</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="${ssrRenderStyle({ "counter-reset": "line-number 0" })}"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div></div><h2 id="实践项目" tabindex="-1"><a class="header-anchor" href="#实践项目"><span>实践项目</span></a></h2><h3 id="_1-数据预处理实战" tabindex="-1"><a class="header-anchor" href="#_1-数据预处理实战"><span>1. 数据预处理实战</span></a></h3><p>本实践项目将带你完成一个完整的数据预处理流程，包括：</p><ol><li><p>数据加载与探索</p><ul><li>使用 Pandas 读取数据</li><li>查看数据基本信息</li><li>数据可视化分析</li></ul></li><li><p>缺失值处理</p><ul><li>检测缺失值</li><li>不同类型特征的填充策略</li><li>缺失值处理效果评估</li></ul></li><li><p>异常值检测</p><ul><li>箱线图分析</li><li>IQR方法检测异常值</li><li>异常值处理策略</li></ul></li><li><p>特征工程</p><ul><li>特征创建与转换</li><li>类别特征编码</li><li>数值特征标准化</li></ul></li><li><p>特征选择</p><ul><li>相关性分析</li><li>特征重要性评估</li><li>降维技术应用</li></ul></li></ol><h3 id="项目要求" tabindex="-1"><a class="header-anchor" href="#项目要求"><span>项目要求</span></a></h3><ol><li><p>环境配置</p><ul><li>Python 3.8+</li><li>pandas, numpy, matplotlib, seaborn</li><li>scikit-learn</li></ul></li><li><p>数据集</p><ul><li>房价预测数据集</li><li>包含数值和类别特征</li><li>存在缺失值和异常值</li></ul></li><li><p>完成任务</p><ul><li>按照notebook中的步骤完成所有代码练习</li><li>理解每个步骤的原理和作用</li><li>尝试使用不同的参数和方法</li></ul></li></ol><h3 id="实践环境" tabindex="-1"><a class="header-anchor" href="#实践环境"><span>实践环境</span></a></h3>`);
  _push(ssrRenderComponent(_component_ClientOnly, null, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(ssrRenderComponent(_component_practice_notebook, {
          title: "数据预处理实战",
          notebook: "https://www.heywhale.com/mw/project/67d20eab372b07bacb11d4ea?shareby=67d20e0833a93c9ff914335e#",
          gitee: "https://gitee.com/ffeng1271383559/datamining-practice/blob/master/notebooks/数据预处理实战.ipynb",
          download: "/notebooks/数据预处理实战.ipynb"
        }, {
          default: withCtx((_2, _push3, _parent3, _scopeId2) => {
            if (_push3) {
              _push3(`<pre${_scopeId2}><code${_scopeId2}># 数据加载与预处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv(&#39;house_prices.csv&#39;)

# 查看缺失值情况
missing = df.isnull().sum()
missing_percent = missing / len(df) * 100

# 处理缺失值
numeric_cols = df.select_dtypes(include=[&#39;int64&#39;, &#39;float64&#39;]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
</code></pre>`);
            } else {
              return [
                createVNode("pre", null, [
                  createVNode("code", null, "# 数据加载与预处理\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# 加载数据\ndf = pd.read_csv('house_prices.csv')\n\n# 查看缺失值情况\nmissing = df.isnull().sum()\nmissing_percent = missing / len(df) * 100\n\n# 处理缺失值\nnumeric_cols = df.select_dtypes(include=['int64', 'float64']).columns\ndf[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())\n")
                ])
              ];
            }
          }),
          _: 1
        }, _parent2, _scopeId));
      } else {
        return [
          createVNode(_component_practice_notebook, {
            title: "数据预处理实战",
            notebook: "https://www.heywhale.com/mw/project/67d20eab372b07bacb11d4ea?shareby=67d20e0833a93c9ff914335e#",
            gitee: "https://gitee.com/ffeng1271383559/datamining-practice/blob/master/notebooks/数据预处理实战.ipynb",
            download: "/notebooks/数据预处理实战.ipynb"
          }, {
            default: withCtx(() => [
              createVNode("pre", null, [
                createVNode("code", null, "# 数据加载与预处理\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# 加载数据\ndf = pd.read_csv('house_prices.csv')\n\n# 查看缺失值情况\nmissing = df.isnull().sum()\nmissing_percent = missing / len(df) * 100\n\n# 处理缺失值\nnumeric_cols = df.select_dtypes(include=['int64', 'float64']).columns\ndf[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())\n")
              ])
            ]),
            _: 1
          })
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`<h3 id="_2-分类算法实践" tabindex="-1"><a class="header-anchor" href="#_2-分类算法实践"><span>2. 分类算法实践</span></a></h3>`);
  _push(ssrRenderComponent(_component_ClientOnly, null, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(ssrRenderComponent(_component_practice_notebook, {
          title: "分类算法实践",
          notebook: "https://aistudio.baidu.com/notebook链接",
          gitee: "https://gitee.com/你的仓库/分类算法.ipynb",
          download: "/notebooks/分类算法实践.ipynb"
        }, null, _parent2, _scopeId));
      } else {
        return [
          createVNode(_component_practice_notebook, {
            title: "分类算法实践",
            notebook: "https://aistudio.baidu.com/notebook链接",
            gitee: "https://gitee.com/你的仓库/分类算法.ipynb",
            download: "/notebooks/分类算法实践.ipynb"
          })
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`</div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/practice/index.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const index_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "index.html.vue"]]);
const data = JSON.parse('{"path":"/practice/","title":"实践指南","lang":"zh-CN","frontmatter":{"title":"实践指南"},"headers":[{"level":2,"title":"在线实践环境","slug":"在线实践环境","link":"#在线实践环境","children":[]},{"level":2,"title":"本地环境配置","slug":"本地环境配置","link":"#本地环境配置","children":[]},{"level":2,"title":"实践项目","slug":"实践项目","link":"#实践项目","children":[{"level":3,"title":"1. 数据预处理实战","slug":"_1-数据预处理实战","link":"#_1-数据预处理实战","children":[]},{"level":3,"title":"项目要求","slug":"项目要求","link":"#项目要求","children":[]},{"level":3,"title":"实践环境","slug":"实践环境","link":"#实践环境","children":[]},{"level":3,"title":"2. 分类算法实践","slug":"_2-分类算法实践","link":"#_2-分类算法实践","children":[]}]}],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"practice/README.md"}');
export {
  index_html as comp,
  data
};
