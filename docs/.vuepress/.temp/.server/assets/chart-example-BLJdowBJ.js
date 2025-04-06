import { resolveComponent, withCtx, createVNode, ref, onMounted, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderComponent, ssrRenderStyle } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  __name: "chart-example",
  setup(__props, { expose: __expose }) {
    __expose();
    const chartRef = ref(null);
    onMounted(async () => {
      if (typeof window !== "undefined") {
        const echarts = await import("echarts");
        const myChart = echarts.init(chartRef.value);
        const option = {
          title: { text: "静态数据折线图" },
          tooltip: { trigger: "axis" },
          // 鼠标悬停显示数据
          toolbox: { feature: { saveAsImage: {} } },
          // 保存图片
          xAxis: { type: "category", data: ["A", "B", "C", "D", "E"] },
          yAxis: { type: "value" },
          series: [{ data: [10, 20, 30, 40, 50], type: "line" }]
        };
        myChart.setOption(option);
      }
    });
    const __returned__ = { chartRef, ref, onMounted };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  const _component_ClientOnly = resolveComponent("ClientOnly");
  _push(`<div${ssrRenderAttrs(_attrs)}><h4>图形展示</h4>`);
  _push(ssrRenderComponent(_component_ClientOnly, null, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`<div style="${ssrRenderStyle({ "width": "600px", "height": "400px" })}"${_scopeId}></div>`);
      } else {
        return [
          createVNode("div", {
            ref: "chartRef",
            style: { "width": "600px", "height": "400px" }
          }, null, 512)
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`</div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/chart-example.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const chartExample = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "chart-example.vue"]]);
export {
  chartExample as default
};
