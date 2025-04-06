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
      resources: [
        {
          type: "tutorial",
          typeText: "教程",
          title: "数据预处理实战指南",
          description: "学习如何处理缺失值、异常值和不一致数据，为分析做好准备。",
          link: "/core/preprocessing/data-presentation.html"
        },
        {
          type: "project",
          typeText: "项目",
          title: "泰坦尼克号生存预测",
          description: "通过分类算法预测乘客生存概率，实践特征工程和模型评估。",
          link: "/projects/classification/titanic.html"
        },
        {
          type: "tool",
          typeText: "工具",
          title: "数据挖掘工具集",
          description: "精选的数据挖掘工具和库，帮助你高效完成分析任务。",
          link: "/resources/tools.html"
        }
      ]
    };
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "resource-cards" }, _attrs))} data-v-5afee4e0><!--[-->`);
  ssrRenderList($data.resources, (resource, index) => {
    _push(`<div class="resource-card" data-v-5afee4e0><div class="${ssrRenderClass([resource.type, "resource-card__type"])}" data-v-5afee4e0>${ssrInterpolate(resource.typeText)}</div><h3 class="resource-card__title" data-v-5afee4e0>${ssrInterpolate(resource.title)}</h3><p class="resource-card__description" data-v-5afee4e0>${ssrInterpolate(resource.description)}</p><a${ssrRenderAttr("href", resource.link)} class="resource-card__link" data-v-5afee4e0>查看详情</a></div>`);
  });
  _push(`<!--]--></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/resource-cards.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const resourceCards = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-5afee4e0"], ["__file", "resource-cards.vue"]]);
export {
  resourceCards as default
};
