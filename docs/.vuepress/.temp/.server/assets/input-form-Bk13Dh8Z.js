import { mergeProps, ref, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrInterpolate } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  __name: "input-form",
  setup(__props, { expose: __expose }) {
    __expose();
    const data = [10, 20, 30, 40, 50];
    const mean = (data.reduce((a, b) => a + b, 0) / data.length).toFixed(2);
    const std = Math.sqrt(
      data.map((x) => (x - mean) ** 2).reduce((a, b) => a + b) / data.length
    ).toFixed(2);
    const __returned__ = { data, mean, std, ref };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "input-form" }, _attrs))} data-v-19215308><h4 data-v-19215308>静态数据分析</h4><div class="output-section" data-v-19215308><p data-v-19215308>数据：10, 20, 30, 40, 50</p><p data-v-19215308>平均值：${ssrInterpolate($setup.mean)}</p><p data-v-19215308>标准差：${ssrInterpolate($setup.std)}</p></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/input-form.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const inputForm = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-19215308"], ["__file", "input-form.vue"]]);
export {
  inputForm as default
};
