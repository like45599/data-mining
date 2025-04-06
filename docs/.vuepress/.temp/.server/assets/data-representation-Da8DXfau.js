import { resolveComponent, mergeProps, ref, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrIncludeBooleanAttr, ssrLooseContain, ssrLooseEqual, ssrRenderList, ssrInterpolate, ssrRenderComponent } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  __name: "data-representation",
  setup(__props, { expose: __expose }) {
    __expose();
    const selectedMethod = ref("table");
    const tableData = ref([10, 20, 30, 40, 50]);
    const mean = (tableData.value.reduce((a, b) => a + b, 0) / tableData.value.length).toFixed(2);
    const std = Math.sqrt(
      tableData.value.map((x) => (x - mean) ** 2).reduce((a, b) => a + b) / tableData.value.length
    ).toFixed(2);
    const __returned__ = { selectedMethod, tableData, mean, std, ref };
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  const _component_chart_example = resolveComponent("chart-example");
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "data-representation" }, _attrs))} data-v-a80ed00f><h3 data-v-a80ed00f>数据表示方法演示</h3><select class="method-select" data-v-a80ed00f><option value="table" data-v-a80ed00f${ssrIncludeBooleanAttr(Array.isArray($setup.selectedMethod) ? ssrLooseContain($setup.selectedMethod, "table") : ssrLooseEqual($setup.selectedMethod, "table")) ? " selected" : ""}>表格形式</option><option value="graph" data-v-a80ed00f${ssrIncludeBooleanAttr(Array.isArray($setup.selectedMethod) ? ssrLooseContain($setup.selectedMethod, "graph") : ssrLooseEqual($setup.selectedMethod, "graph")) ? " selected" : ""}>图形可视化</option><option value="math" data-v-a80ed00f${ssrIncludeBooleanAttr(Array.isArray($setup.selectedMethod) ? ssrLooseContain($setup.selectedMethod, "math") : ssrLooseEqual($setup.selectedMethod, "math")) ? " selected" : ""}>数学表达式</option></select>`);
  if ($setup.selectedMethod === "table") {
    _push(`<div class="table-view" data-v-a80ed00f><h4 data-v-a80ed00f>表格形式展示</h4><table data-v-a80ed00f><thead data-v-a80ed00f><tr data-v-a80ed00f><th data-v-a80ed00f>数据点</th><th data-v-a80ed00f>值</th></tr></thead><tbody data-v-a80ed00f><!--[-->`);
    ssrRenderList($setup.tableData, (value, index) => {
      _push(`<tr data-v-a80ed00f><td data-v-a80ed00f>${ssrInterpolate(index + 1)}</td><td data-v-a80ed00f>${ssrInterpolate(value)}</td></tr>`);
    });
    _push(`<!--]--></tbody></table></div>`);
  } else {
    _push(`<!---->`);
  }
  if ($setup.selectedMethod === "graph") {
    _push(`<div class="graph-view" data-v-a80ed00f>`);
    _push(ssrRenderComponent(_component_chart_example, null, null, _parent));
    _push(`</div>`);
  } else {
    _push(`<!---->`);
  }
  if ($setup.selectedMethod === "math") {
    _push(`<div class="math-view" data-v-a80ed00f><h4 data-v-a80ed00f>数学表达式展示</h4><p data-v-a80ed00f>平均值：${ssrInterpolate($setup.mean)}</p><p data-v-a80ed00f>标准差：${ssrInterpolate($setup.std)}</p></div>`);
  } else {
    _push(`<!---->`);
  }
  _push(`</div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/data-representation.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const dataRepresentation = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-a80ed00f"], ["__file", "data-representation.vue"]]);
export {
  dataRepresentation as default
};
