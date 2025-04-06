import { mergeProps, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrInterpolate, ssrRenderList } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  name: "DisciplineMap",
  computed: {
    isEnglish() {
      return this.$lang === "en-US";
    },
    disciplines() {
      if (this.isEnglish) {
        return [
          {
            icon: "📊",
            name: "Statistics",
            description: "Provides theoretical foundation and methodology for data analysis"
          },
          {
            icon: "🧠",
            name: "Machine Learning",
            description: "Provides algorithms and models for automated learning and prediction"
          },
          {
            icon: "💾",
            name: "Database Technology",
            description: "Provides methods for data storage, querying and management"
          },
          {
            icon: "📈",
            name: "Business Intelligence",
            description: "Applies data mining results to support business decisions"
          },
          {
            icon: "🔮",
            name: "Artificial Intelligence",
            description: "Provides intelligent systems and complex problem-solving methods"
          },
          {
            icon: "🖥️",
            name: "Computer Science",
            description: "Provides algorithm design and computational foundations"
          }
        ];
      } else {
        return [
          {
            icon: "📊",
            name: "统计学",
            description: "提供数据分析的理论基础和方法论"
          },
          {
            icon: "🧠",
            name: "机器学习",
            description: "提供自动化学习和预测的算法和模型"
          },
          {
            icon: "💾",
            name: "数据库技术",
            description: "提供数据存储、查询和管理的方法"
          },
          {
            icon: "📈",
            name: "商业智能",
            description: "应用数据挖掘结果支持业务决策"
          },
          {
            icon: "🔮",
            name: "人工智能",
            description: "提供智能系统和复杂问题求解方法"
          },
          {
            icon: "🖥️",
            name: "计算机科学",
            description: "提供算法设计和计算基础"
          }
        ];
      }
    }
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "discipline-map" }, _attrs))} data-v-ebf3ce8b><div class="discipline-center" data-v-ebf3ce8b><div class="discipline-icon" data-v-ebf3ce8b>🔍</div><div class="discipline-name" data-v-ebf3ce8b>${ssrInterpolate($options.isEnglish ? "Data Mining" : "数据挖掘")}</div></div><div class="discipline-connections" data-v-ebf3ce8b><!--[-->`);
  ssrRenderList($options.disciplines, (discipline, index) => {
    _push(`<div class="discipline-item" data-v-ebf3ce8b><div class="discipline-icon" data-v-ebf3ce8b>${ssrInterpolate(discipline.icon)}</div><div class="discipline-name" data-v-ebf3ce8b>${ssrInterpolate(discipline.name)}</div><div class="discipline-desc" data-v-ebf3ce8b>${ssrInterpolate(discipline.description)}</div></div>`);
  });
  _push(`<!--]--></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/DisciplineMap.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const DisciplineMap = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-ebf3ce8b"], ["__file", "DisciplineMap.vue"]]);
export {
  DisciplineMap as default
};
