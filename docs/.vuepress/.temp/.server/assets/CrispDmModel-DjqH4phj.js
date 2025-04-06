import { mergeProps, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrInterpolate, ssrRenderList } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  name: "CrispDmModel",
  computed: {
    isEnglish() {
      return this.$lang === "en-US";
    },
    phases() {
      if (this.isEnglish) {
        return [
          {
            title: "Business Understanding",
            description: "Determine business objectives, convert them into data mining problems, and develop initial plan",
            tasks: [
              "Define business objectives",
              "Assess situation",
              "Determine data mining goals",
              "Produce project plan"
            ]
          },
          {
            title: "Data Understanding",
            description: "Collect initial data, get familiar with data, identify data quality issues, discover insights",
            tasks: [
              "Collect initial data",
              "Describe data",
              "Explore data",
              "Verify data quality"
            ]
          },
          {
            title: "Data Preparation",
            description: "Select, clean, construct, integrate and format data for modeling",
            tasks: [
              "Select data",
              "Clean data",
              "Construct data",
              "Integrate data",
              "Format data"
            ]
          },
          {
            title: "Modeling",
            description: "Select modeling techniques, generate test design, build and assess models",
            tasks: [
              "Select modeling techniques",
              "Generate test design",
              "Build models",
              "Assess models"
            ]
          },
          {
            title: "Evaluation",
            description: "Evaluate results, review process, determine next steps",
            tasks: [
              "Evaluate results",
              "Review process",
              "Determine next steps"
            ]
          },
          {
            title: "Deployment",
            description: "Plan deployment, monitoring and maintenance, produce final report, review project",
            tasks: [
              "Plan deployment",
              "Plan monitoring",
              "Produce final report",
              "Review project"
            ]
          }
        ];
      } else {
        return [
          {
            title: "业务理解",
            description: "确定业务目标，将其转化为数据挖掘问题，并制定初步计划",
            tasks: [
              "确定业务目标",
              "评估现状",
              "确定数据挖掘目标",
              "制定项目计划"
            ]
          },
          {
            title: "数据理解",
            description: "收集初始数据，熟悉数据，识别数据质量问题，发现数据洞察",
            tasks: [
              "收集初始数据",
              "描述数据",
              "探索数据",
              "验证数据质量"
            ]
          },
          {
            title: "数据准备",
            description: "选择、清洗、构建、集成和格式化数据以供建模使用",
            tasks: [
              "选择数据",
              "清洗数据",
              "构建数据",
              "集成数据",
              "格式化数据"
            ]
          },
          {
            title: "建模",
            description: "选择建模技术，生成测试设计，构建和评估模型",
            tasks: [
              "选择建模技术",
              "生成测试设计",
              "构建模型",
              "评估模型"
            ]
          },
          {
            title: "评估",
            description: "评估结果，审查过程，确定下一步",
            tasks: [
              "评估结果",
              "审查过程",
              "确定下一步"
            ]
          },
          {
            title: "部署",
            description: "规划部署、监控和维护，生成最终报告，回顾项目",
            tasks: [
              "规划部署",
              "规划监控",
              "生成最终报告",
              "回顾项目"
            ]
          }
        ];
      }
    }
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "crisp-dm-container" }, _attrs))} data-v-a3dae1ad><div class="crisp-dm-center" data-v-a3dae1ad><div class="crisp-dm-icon" data-v-a3dae1ad>🔄</div><div class="crisp-dm-title" data-v-a3dae1ad>CRISP-DM</div><div class="crisp-dm-subtitle" data-v-a3dae1ad>${ssrInterpolate($options.isEnglish ? "Cross-Industry Standard Process for Data Mining" : "跨行业标准数据挖掘流程")}</div></div><div class="crisp-dm-phases" data-v-a3dae1ad><!--[-->`);
  ssrRenderList($options.phases, (phase, index) => {
    _push(`<div class="crisp-dm-phase" data-v-a3dae1ad><div class="phase-number" data-v-a3dae1ad>${ssrInterpolate(index + 1)}</div><div class="phase-content" data-v-a3dae1ad><div class="phase-title" data-v-a3dae1ad>${ssrInterpolate(phase.title)}</div><div class="phase-desc" data-v-a3dae1ad>${ssrInterpolate(phase.description)}</div><div class="phase-tasks" data-v-a3dae1ad><!--[-->`);
    ssrRenderList(phase.tasks, (task, taskIndex) => {
      _push(`<span class="task-item" data-v-a3dae1ad>${ssrInterpolate(task)}</span>`);
    });
    _push(`<!--]--></div></div></div>`);
  });
  _push(`<!--]--></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/CrispDmModel.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const CrispDmModel = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-a3dae1ad"], ["__file", "CrispDmModel.vue"]]);
export {
  CrispDmModel as default
};
