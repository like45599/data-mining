import { ssrRenderAttrs } from "vue/server-renderer";
import { useSSRContext } from "vue";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
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
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/index.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const index_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "index.html.vue"]]);
const data = JSON.parse('{"path":"/","title":"","lang":"zh-CN","frontmatter":{"home":true,"heroImage":"/images/index.jpg","heroText":"数据挖掘：从理论到实践的探索之旅","tagline":"系统学习数据挖掘的核心概念与实践技能","actions":[{"text":"开始学习","link":"/overview/","type":"primary"},{"text":"实践项目","link":"/projects/","type":"secondary"}],"features":[{"title":"数据预处理","details":"学习数据表示、缺失值处理、异常值检测等数据准备技术，为后续分析打下坚实基础。"},{"title":"分类算法","details":"掌握SVM、朴素贝叶斯、决策树等主流分类算法，学会解决多种实际分类问题。"},{"title":"聚类分析","details":"理解K-Means等聚类算法原理，学习无监督学习在客户分群等场景中的应用。"},{"title":"预测与回归","details":"学习回归分析方法，构建预测模型，并掌握模型评估与优化的关键技术。"}],"footer":"Created by E222"},"headers":[],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"README.md"}');
export {
  index_html as comp,
  data
};
