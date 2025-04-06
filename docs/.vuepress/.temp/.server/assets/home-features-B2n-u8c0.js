import { mergeProps, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderList, ssrRenderClass, ssrInterpolate, ssrRenderAttr } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  data() {
    return {
      features: [
        {
          title: "数据预处理",
          description: "学习如何清洗、转换和准备数据，为后续分析奠定基础。",
          icon: "icon-data",
          link: "/core/preprocessing/data-presentation.html"
        },
        {
          title: "分类算法",
          description: "掌握SVM、朴素贝叶斯和决策树等主要分类方法。",
          icon: "icon-classify",
          link: "/core/classification/svm.html"
        },
        {
          title: "聚类分析",
          description: "探索K-Means等无监督学习方法，发现数据中的自然分组。",
          icon: "icon-cluster",
          link: "/core/clustering/kmeans.html"
        },
        {
          title: "预测与回归",
          description: "学习如何使用线性和非线性回归方法进行预测分析。",
          icon: "icon-predict",
          link: "/core/regression/linear-regression.html"
        }
      ]
    };
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "features-container" }, _attrs))} data-v-cb063600><!--[-->`);
  ssrRenderList($data.features, (feature, index) => {
    _push(`<div class="feature-card" data-v-cb063600><div class="feature-icon" data-v-cb063600><i class="${ssrRenderClass(feature.icon)}" data-v-cb063600></i></div><h3 data-v-cb063600>${ssrInterpolate(feature.title)}</h3><p data-v-cb063600>${ssrInterpolate(feature.description)}</p><a${ssrRenderAttr("href", feature.link)} class="feature-link" data-v-cb063600>了解更多</a></div>`);
  });
  _push(`<!--]--></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/home-features.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const homeFeatures = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-cb063600"], ["__file", "home-features.vue"]]);
export {
  homeFeatures as default
};
