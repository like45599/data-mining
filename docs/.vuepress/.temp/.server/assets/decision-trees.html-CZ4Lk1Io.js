import { ssrRenderAttrs } from "vue/server-renderer";
import { useSSRContext } from "vue";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="决策树" tabindex="-1"><a class="header-anchor" href="#决策树"><span>决策树</span></a></h1><h2 id="基本概念" tabindex="-1"><a class="header-anchor" href="#基本概念"><span>基本概念</span></a></h2><p>决策树是一种树形结构的分类模型，通过一系列规则对数据进行分类。</p><h2 id="构建算法" tabindex="-1"><a class="header-anchor" href="#构建算法"><span>构建算法</span></a></h2><ol><li>ID3算法</li><li>C4.5算法</li><li>CART算法</li></ol><h2 id="重要概念" tabindex="-1"><a class="header-anchor" href="#重要概念"><span>重要概念</span></a></h2><ol><li>信息增益</li><li>基尼指数</li><li>剪枝策略</li></ol><h2 id="实践案例" tabindex="-1"><a class="header-anchor" href="#实践案例"><span>实践案例</span></a></h2><p>[待补充具体案例代码和数据]</p></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/guide/decision-trees.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const decisionTrees_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "decision-trees.html.vue"]]);
const data = JSON.parse('{"path":"/guide/decision-trees.html","title":"决策树","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"基本概念","slug":"基本概念","link":"#基本概念","children":[]},{"level":2,"title":"构建算法","slug":"构建算法","link":"#构建算法","children":[]},{"level":2,"title":"重要概念","slug":"重要概念","link":"#重要概念","children":[]},{"level":2,"title":"实践案例","slug":"实践案例","link":"#实践案例","children":[]}],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"guide/decision-trees.md"}');
export {
  decisionTrees_html as comp,
  data
};
