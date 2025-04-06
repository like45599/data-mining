import { mergeProps, ref, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderList, ssrInterpolate, ssrRenderAttr } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  __name: "ppt-resources",
  setup(__props, { expose: __expose }) {
    __expose();
    const pptList = ref([
      {
        title: "L1 数据表示与预处理",
        description: "介绍数据类型、特征工程、数据清洗等基础知识，包含pandas实战示例",
        link: encodeURI("/resources/ppt/L1 Data presentation.pptx"),
        keywords: ["数据类型", "特征工程", "pandas", "数据清洗"]
      },
      {
        title: "L2 缺失值处理",
        description: "详解各种缺失值处理策略，包含实际案例分析和最佳实践指南",
        link: encodeURI("/resources/ppt/L2 Restoring missing values.pptx"),
        keywords: ["缺失值类型", "填充策略", "删除策略", "高级处理方法"]
      },
      {
        title: "L3 分类与SVM",
        description: "SVM分类器原理与应用",
        link: encodeURI("/resources/ppt/L3 Classification. SVM.pptx")
      },
      {
        title: "L4 垃圾邮件过滤",
        description: "基于朴素贝叶斯的文本分类",
        link: encodeURI("/resources/ppt/L4 Spam filters. Naive Baies.pptx")
      },
      {
        title: "L5 决策树",
        description: "决策树算法与应用",
        link: encodeURI("/resources/ppt/L5 Decision trees.pptx")
      },
      {
        title: "L6 聚类分析",
        description: "K-Means聚类算法",
        link: encodeURI("/resources/ppt/L6  Clusterization. K-Means.pptx")
      },
      {
        title: "L7 预测与回归",
        description: "回归分析方法",
        link: encodeURI("/resources/ppt/L7 Prediction. Regression.pptx")
      }
    ]);
    const __returned__ = { pptList, ref };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "resource-container" }, _attrs))} data-v-7719e177><div class="resource-grid" data-v-7719e177><!--[-->`);
  ssrRenderList($setup.pptList, (ppt, index) => {
    _push(`<div class="resource-card" data-v-7719e177><div class="resource-icon" data-v-7719e177><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512" class="icon-ppt" data-v-7719e177><path d="M256 0v128h128L256 0zM224 128L224 0H48C21.49 0 0 21.49 0 48v416C0 490.5 21.49 512 48 512h288c26.51 0 48-21.49 48-48V160h-127.1C238.3 160 224 145.7 224 128z" data-v-7719e177></path></svg></div><div class="resource-info" data-v-7719e177><h3 data-v-7719e177>${ssrInterpolate(ppt.title)}</h3><p data-v-7719e177>${ssrInterpolate(ppt.description)}</p><a${ssrRenderAttr("href", ppt.link)} class="download-btn" download data-v-7719e177> 下载 </a></div></div>`);
  });
  _push(`<!--]--></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/ppt-resources.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const pptResources = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-7719e177"], ["__file", "ppt-resources.vue"]]);
export {
  pptResources as default
};
