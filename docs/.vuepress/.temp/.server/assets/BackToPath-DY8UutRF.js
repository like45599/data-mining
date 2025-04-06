import { mergeProps, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderAttr, ssrInterpolate } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  name: "BackToPath",
  data() {
    return {
      showBackButton: false
    };
  },
  computed: {
    buttonText() {
      return this.$lang === "en-US" ? "Back to Learning Path" : "返回学习路径";
    },
    pathPrefix() {
      return this.$lang === "en-US" ? "/en" : "";
    }
  },
  mounted() {
    const referrer = document.referrer;
    const currentHost = window.location.host;
    if (referrer.includes(currentHost) && (referrer.includes("/learning-path/") || referrer.includes("/en/learning-path/"))) {
      this.showBackButton = true;
    }
    if (localStorage.getItem("fromLearningPath") === "true") {
      this.showBackButton = true;
      localStorage.removeItem("fromLearningPath");
    }
  },
  created() {
    this.$nextTick(() => {
      const learningPathLinks = document.querySelectorAll('a[href*="/learning-path/"]');
      learningPathLinks.forEach((link) => {
        link.addEventListener("click", () => {
          sessionStorage.setItem("fromLearningPath", "true");
        });
      });
    });
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  if ($data.showBackButton) {
    _push(`<div${ssrRenderAttrs(mergeProps({ class: "back-to-path" }, _attrs))} data-v-903e279e><a${ssrRenderAttr("href", $options.pathPrefix + "/learning-path/")} class="back-to-path__button" data-v-903e279e><span class="back-to-path__icon" data-v-903e279e>←</span> ${ssrInterpolate($options.buttonText)}</a></div>`);
  } else {
    _push(`<!---->`);
  }
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/BackToPath.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const BackToPath = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-903e279e"], ["__file", "BackToPath.vue"]]);
export {
  BackToPath as default
};
