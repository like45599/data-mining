import { ssrRenderAttrs } from "vue/server-renderer";
import { useSSRContext } from "vue";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  _push(`<div${ssrRenderAttrs(_attrs)}></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/en/index.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const index_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "index.html.vue"]]);
const data = JSON.parse('{"path":"/en/","title":"","lang":"en-US","frontmatter":{"home":true,"heroImage":"/images/index.jpg","heroText":"Data Mining Learning Guide","tagline":"A comprehensive guide to data mining concepts and techniques","actions":[{"text":"Get Started →","link":"/en/overview/","type":"primary"},{"text":"Learning Path","link":"/en/learning-path/","type":"secondary"}],"features":[{"title":"data preprocessing","details":"Learn data preparation techniques such as data representation, missing value processing, outlier detection, etc., to lay a solid foundation for subsequent analysis."},{"title":"indicates the classification algorithm","details":"Master SVM, naive Bayes, decision tree and other mainstream classification algorithms, learn to solve a variety of practical classification problems."},{"title":"cluster analysis","details":"Understand the principle of clustering algorithm such as K-Means, and learn the application of unsupervised learning in scenarios such as customer clustering."},{"title":"prediction and regression","details":"Learn regression analysis methods, build prediction models, and master key techniques of model evaluation and optimization."}],"footer":"E222"},"headers":[],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"en/README.md"}');
export {
  index_html as comp,
  data
};
