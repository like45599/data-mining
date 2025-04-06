import { mergeProps, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrInterpolate, ssrRenderAttr, ssrRenderSlot } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  __name: "practice-notebook",
  props: {
    title: String,
    notebook: String,
    gitee: String,
    download: String
  },
  setup(__props, { expose: __expose }) {
    __expose();
    const __returned__ = {};
    Object.defineProperty(__returned__, "__isScriptSetup", { enumerable: false, value: true });
    return __returned__;
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "notebook-container" }, _attrs))} data-v-4f4c4989><div class="notebook-card" data-v-4f4c4989><h3 data-v-4f4c4989>${ssrInterpolate($props.title)}</h3><div class="notebook-buttons" data-v-4f4c4989><a${ssrRenderAttr("href", $props.notebook)} target="_blank" class="button notebook-button" data-v-4f4c4989><svg class="icon" viewBox="0 0 1024 1024" data-v-4f4c4989><path d="M512 0c282.784 0 512 229.216 512 512s-229.216 512-512 512S0 794.784 0 512 229.216 0 512 0z" fill="#F37626" data-v-4f4c4989></path><path d="M..." fill="#fff" data-v-4f4c4989></path></svg> 在线运行 </a><a${ssrRenderAttr("href", $props.gitee)} target="_blank" class="button gitee-button" data-v-4f4c4989><svg class="icon" viewBox="0 0 1024 1024" data-v-4f4c4989><path d="M512 0c282.784 0 512 229.216 512 512s-229.216 512-512 512S0 794.784 0 512 229.216 0 512 0z" fill="#C71D23" data-v-4f4c4989></path><path d="M..." fill="#fff" data-v-4f4c4989></path></svg> 查看代码 </a><a${ssrRenderAttr("href", $props.download)} class="button download-button" download data-v-4f4c4989><svg class="icon" viewBox="0 0 1024 1024" data-v-4f4c4989><path d="M..." fill="currentColor" data-v-4f4c4989></path></svg> 下载文件 </a></div><div class="notebook-preview" data-v-4f4c4989><div class="preview-header" data-v-4f4c4989><span class="preview-title" data-v-4f4c4989>代码预览</span><span class="preview-info" data-v-4f4c4989>建议下载后在本地运行完整版本</span></div><div class="preview-content" data-v-4f4c4989>`);
  ssrRenderSlot(_ctx.$slots, "default", {}, null, _push, _parent);
  _push(`</div></div></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/practice-notebook.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const practiceNotebook = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-4f4c4989"], ["__file", "practice-notebook.vue"]]);
export {
  practiceNotebook as default
};
