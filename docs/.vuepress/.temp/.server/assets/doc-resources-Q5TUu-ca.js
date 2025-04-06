import { mergeProps, ref, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderList, ssrInterpolate, ssrRenderAttr } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  __name: "doc-resources",
  setup(__props, { expose: __expose }) {
    __expose();
    const docList = ref([
      {
        title: "数据预处理实践任务作业",
        description: "包含数据清洗和特征工程的实践任务",
        link: encodeURI("/resources/docs/П1.docx")
      },
      {
        title: "SVM分类器实现任务作业",
        description: "基于Python的SVM分类器实现",
        link: encodeURI("/resources/docs/П2.docx")
      },
      {
        title: "朴素贝叶斯实现任务",
        description: "垃圾邮件分类器的实现指南",
        link: encodeURI("/resources/docs/П3.docx")
      },
      {
        title: "决策树模型实践",
        description: "使用决策树解决分类问题",
        link: encodeURI("/resources/docs/П4.docx")
      },
      {
        title: "K-Means聚类分析",
        description: "客户分群案例实践",
        link: encodeURI("/resources/docs/П5.docx")
      },
      {
        title: "回归分析实践",
        description: "预测模型的构建与评估",
        link: encodeURI("/resources/docs/П6.docx")
      },
      {
        title: "综合项目实践",
        description: "数据挖掘方法的综合应用",
        link: encodeURI("/resources/docs/П7.docx")
      }
    ]);
    const __returned__ = { docList, ref };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "resource-container" }, _attrs))} data-v-5b0586ee><div class="resource-grid" data-v-5b0586ee><!--[-->`);
  ssrRenderList($setup.docList, (doc, index) => {
    _push(`<div class="resource-card" data-v-5b0586ee><div class="resource-icon" data-v-5b0586ee><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512" class="icon-doc" data-v-5b0586ee><path d="M365.3 93.38l-74.63-74.64C278.6 6.742 262.3 0 245.4 0H64C28.65 0 0 28.65 0 64l.0065 384c0 35.34 28.65 64 64 64H320c35.2 0 64-28.8 64-64V138.6C384 121.7 377.3 105.4 365.3 93.38zM336 448c0 8.836-7.164 16-16 16H64.02c-8.838 0-16-7.164-16-16L48 64.13c0-8.836 7.164-16 16-16h160L224 128c0 17.67 14.33 32 32 32h79.1V448z" data-v-5b0586ee></path></svg></div><div class="resource-info" data-v-5b0586ee><h3 data-v-5b0586ee>${ssrInterpolate(doc.title)}</h3><p data-v-5b0586ee>${ssrInterpolate(doc.description)}</p><a${ssrRenderAttr("href", doc.link)} class="download-btn" download data-v-5b0586ee> 下载 </a></div></div>`);
  });
  _push(`<!--]--></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/doc-resources.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const docResources = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-5b0586ee"], ["__file", "doc-resources.vue"]]);
export {
  docResources as default
};
