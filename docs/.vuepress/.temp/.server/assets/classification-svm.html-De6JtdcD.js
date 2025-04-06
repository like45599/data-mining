import { ssrRenderAttrs } from "vue/server-renderer";
import { useSSRContext } from "vue";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="支持向量机分类" tabindex="-1"><a class="header-anchor" href="#支持向量机分类"><span>支持向量机分类</span></a></h1><h2 id="基本概念" tabindex="-1"><a class="header-anchor" href="#基本概念"><span>基本概念</span></a></h2><p>支持向量机（SVM）是一种强大的监督学习模型，主要用于分类问题。它通过寻找最优超平面来实现数据分类。</p><h2 id="核心原理" tabindex="-1"><a class="header-anchor" href="#核心原理"><span>核心原理</span></a></h2><ol><li>最大间隔超平面</li><li>核函数技巧</li><li>软间隔</li></ol><h2 id="常用核函数" tabindex="-1"><a class="header-anchor" href="#常用核函数"><span>常用核函数</span></a></h2><ol><li>线性核</li><li>多项式核</li><li>RBF核（高斯核）</li><li>Sigmoid核</li></ol><h2 id="实践案例" tabindex="-1"><a class="header-anchor" href="#实践案例"><span>实践案例</span></a></h2><p>[待补充具体案例代码和数据]</p></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/guide/classification-svm.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const classificationSvm_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "classification-svm.html.vue"]]);
const data = JSON.parse('{"path":"/guide/classification-svm.html","title":"支持向量机分类","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"基本概念","slug":"基本概念","link":"#基本概念","children":[]},{"level":2,"title":"核心原理","slug":"核心原理","link":"#核心原理","children":[]},{"level":2,"title":"常用核函数","slug":"常用核函数","link":"#常用核函数","children":[]},{"level":2,"title":"实践案例","slug":"实践案例","link":"#实践案例","children":[]}],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"guide/classification-svm.md"}');
export {
  classificationSvm_html as comp,
  data
};
