import { resolveComponent, withCtx, createTextVNode, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderComponent } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  const _component_RouteLink = resolveComponent("RouteLink");
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="数据预处理项目" tabindex="-1"><a class="header-anchor" href="#数据预处理项目"><span>数据预处理项目</span></a></h1><div class="knowledge-card"><div class="knowledge-card__title"><span class="icon">📚</span>本节要点 </div><div class="knowledge-card__content"><ul><li>了解数据预处理在数据挖掘中的重要性</li><li>掌握处理缺失值、异常值的实用技术</li><li>学习特征工程的基本方法</li><li>通过实践项目应用数据预处理技术</li></ul></div></div><h2 id="数据预处理的重要性" tabindex="-1"><a class="header-anchor" href="#数据预处理的重要性"><span>数据预处理的重要性</span></a></h2><p>数据预处理是数据挖掘过程中最关键的步骤之一，通常占据整个项目时间的60-70%。高质量的数据预处理可以显著提高模型性能，而忽视这一步骤则可能导致&quot;垃圾进，垃圾出&quot;的结果。</p><p>数据预处理主要解决以下问题：</p><ul><li>处理缺失值和异常值</li><li>标准化和归一化数据</li><li>转换数据格式</li><li>创建和选择有效特征</li><li>降维和数据平衡</li></ul><h2 id="预处理项目列表" tabindex="-1"><a class="header-anchor" href="#预处理项目列表"><span>预处理项目列表</span></a></h2><p>以下项目专注于数据预处理技术的应用，帮助你掌握这一关键技能：</p><h3 id="电商用户数据清洗与分析" tabindex="-1"><a class="header-anchor" href="#电商用户数据清洗与分析"><span>`);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/projects/preprocessing/ecommerce-data.html" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`电商用户数据清洗与分析`);
      } else {
        return [
          createTextVNode("电商用户数据清洗与分析")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`</span></a></h3><div class="project-card"><div class="project-card__header"><div class="project-card__title">电商用户数据清洗与分析</div><div class="project-card__tags"><span class="tag">数据清洗</span><span class="tag">入门级</span></div></div><div class="project-card__content"><p>处理电子商务网站的用户行为数据，包括缺失值处理、异常检测和特征创建。通过这个项目，你将学习如何准备数据用于后续的用户行为分析。</p><div class="project-card__skills"><span class="skill">缺失值处理</span><span class="skill">异常检测</span><span class="skill">数据转换</span><span class="skill">特征创建</span></div></div><div class="project-card__footer"><a href="/projects/preprocessing/ecommerce-data.html" class="button">查看详情</a></div></div><h3 id="医疗数据缺失值处理" tabindex="-1"><a class="header-anchor" href="#医疗数据缺失值处理"><span>`);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/projects/preprocessing/medical-missing-values.html" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`医疗数据缺失值处理`);
      } else {
        return [
          createTextVNode("医疗数据缺失值处理")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`</span></a></h3><div class="project-card"><div class="project-card__header"><div class="project-card__title">医疗数据缺失值处理</div><div class="project-card__tags"><span class="tag">缺失值</span><span class="tag">中级</span></div></div><div class="project-card__content"><p>处理医疗数据集中的缺失值，比较不同缺失值处理方法的效果。这个项目将教你如何在敏感数据中应用高级缺失值处理技术。</p><div class="project-card__skills"><span class="skill">多重插补</span><span class="skill">KNN填充</span><span class="skill">模型预测填充</span><span class="skill">缺失模式分析</span></div></div><div class="project-card__footer"><a href="/projects/preprocessing/medical-missing-values.html" class="button">查看详情</a></div></div><h2 id="数据预处理技能提升" tabindex="-1"><a class="header-anchor" href="#数据预处理技能提升"><span>数据预处理技能提升</span></a></h2><p>通过完成这些项目，你将掌握以下关键技能：</p><h3 id="缺失值处理技术" tabindex="-1"><a class="header-anchor" href="#缺失值处理技术"><span>缺失值处理技术</span></a></h3><ul><li>识别缺失值模式（随机缺失、非随机缺失）</li><li>应用不同的填充方法（均值/中位数/众数填充、KNN填充、模型预测填充）</li><li>评估不同填充方法的影响</li></ul><h3 id="异常值检测与处理" tabindex="-1"><a class="header-anchor" href="#异常值检测与处理"><span>异常值检测与处理</span></a></h3><ul><li>使用统计方法检测异常值（Z-分数、IQR方法）</li><li>应用机器学习方法识别异常（隔离森林、单类SVM）</li><li>选择合适的异常值处理策略</li></ul><h3 id="特征工程基础" tabindex="-1"><a class="header-anchor" href="#特征工程基础"><span>特征工程基础</span></a></h3><ul><li>创建派生特征</li><li>特征编码（独热编码、标签编码、目标编码）</li><li>特征变换（对数变换、Box-Cox变换）</li><li>特征标准化和归一化</li></ul><div class="knowledge-card"><div class="knowledge-card__title"><span class="icon">💡</span>预处理最佳实践 </div><div class="knowledge-card__content"><ul><li><strong>了解你的数据</strong>：在应用任何预处理技术前，先深入理解数据的业务含义</li><li><strong>保留原始数据</strong>：始终保留一份原始数据的副本，以便需要时回溯</li><li><strong>记录所有步骤</strong>：详细记录每个预处理步骤，确保可重复性</li><li><strong>验证结果</strong>：通过可视化和统计检验验证预处理的效果</li><li><strong>考虑业务影响</strong>：评估预处理决策对最终业务目标的影响</li></ul></div></div><h2 id="小结与下一步" tabindex="-1"><a class="header-anchor" href="#小结与下一步"><span>小结与下一步</span></a></h2><p>数据预处理是构建成功数据挖掘项目的基础。通过这些项目，你将学习如何处理真实世界中的数据挑战，为后续的分析和建模奠定坚实基础。</p><h3 id="关键要点回顾" tabindex="-1"><a class="header-anchor" href="#关键要点回顾"><span>关键要点回顾</span></a></h3><ul><li>数据预处理通常占据数据挖掘项目的大部分时间</li><li>高质量的数据预处理可以显著提高模型性能</li><li>不同类型的数据需要不同的预处理策略</li><li>预处理决策应考虑业务背景和后续分析需求</li></ul><h3 id="下一步行动" tabindex="-1"><a class="header-anchor" href="#下一步行动"><span>下一步行动</span></a></h3><ol><li>选择一个预处理项目开始实践</li><li>尝试应用不同的预处理技术并比较结果</li><li>将学到的预处理技能应用到自己的数据集中</li><li>探索更高级的特征工程技术</li></ol><div class="practice-link"><a href="/projects/preprocessing/ecommerce-data.html" class="button">开始第一个预处理项目</a></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/projects/preprocessing.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const preprocessing_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "preprocessing.html.vue"]]);
const data = JSON.parse('{"path":"/projects/preprocessing.html","title":"数据预处理项目","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"数据预处理的重要性","slug":"数据预处理的重要性","link":"#数据预处理的重要性","children":[]},{"level":2,"title":"预处理项目列表","slug":"预处理项目列表","link":"#预处理项目列表","children":[{"level":3,"title":"电商用户数据清洗与分析","slug":"电商用户数据清洗与分析","link":"#电商用户数据清洗与分析","children":[]},{"level":3,"title":"医疗数据缺失值处理","slug":"医疗数据缺失值处理","link":"#医疗数据缺失值处理","children":[]}]},{"level":2,"title":"数据预处理技能提升","slug":"数据预处理技能提升","link":"#数据预处理技能提升","children":[{"level":3,"title":"缺失值处理技术","slug":"缺失值处理技术","link":"#缺失值处理技术","children":[]},{"level":3,"title":"异常值检测与处理","slug":"异常值检测与处理","link":"#异常值检测与处理","children":[]},{"level":3,"title":"特征工程基础","slug":"特征工程基础","link":"#特征工程基础","children":[]}]},{"level":2,"title":"小结与下一步","slug":"小结与下一步","link":"#小结与下一步","children":[{"level":3,"title":"关键要点回顾","slug":"关键要点回顾","link":"#关键要点回顾","children":[]},{"level":3,"title":"下一步行动","slug":"下一步行动","link":"#下一步行动","children":[]}]}],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"projects/preprocessing.md"}');
export {
  preprocessing_html as comp,
  data
};
