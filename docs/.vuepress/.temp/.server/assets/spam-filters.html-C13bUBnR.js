import { ssrRenderAttrs } from "vue/server-renderer";
import { useSSRContext } from "vue";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="垃圾邮件过滤" tabindex="-1"><a class="header-anchor" href="#垃圾邮件过滤"><span>垃圾邮件过滤</span></a></h1><h2 id="朴素贝叶斯原理" tabindex="-1"><a class="header-anchor" href="#朴素贝叶斯原理"><span>朴素贝叶斯原理</span></a></h2><p>朴素贝叶斯是一种基于贝叶斯定理的分类算法，特别适用于文本分类问题。</p><h2 id="文本预处理" tabindex="-1"><a class="header-anchor" href="#文本预处理"><span>文本预处理</span></a></h2><ol><li>分词</li><li>去停用词</li><li>词袋模型</li><li>TF-IDF转换</li></ol><h2 id="实现步骤" tabindex="-1"><a class="header-anchor" href="#实现步骤"><span>实现步骤</span></a></h2><ol><li>数据收集</li><li>特征提取</li><li>模型训练</li><li>性能评估</li></ol><h2 id="实践案例" tabindex="-1"><a class="header-anchor" href="#实践案例"><span>实践案例</span></a></h2><p>[待补充具体案例代码和数据]</p></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/guide/spam-filters.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const spamFilters_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "spam-filters.html.vue"]]);
const data = JSON.parse('{"path":"/guide/spam-filters.html","title":"垃圾邮件过滤","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"朴素贝叶斯原理","slug":"朴素贝叶斯原理","link":"#朴素贝叶斯原理","children":[]},{"level":2,"title":"文本预处理","slug":"文本预处理","link":"#文本预处理","children":[]},{"level":2,"title":"实现步骤","slug":"实现步骤","link":"#实现步骤","children":[]},{"level":2,"title":"实践案例","slug":"实践案例","link":"#实践案例","children":[]}],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"guide/spam-filters.md"}');
export {
  spamFilters_html as comp,
  data
};
