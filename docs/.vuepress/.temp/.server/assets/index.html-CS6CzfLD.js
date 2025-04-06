import { resolveComponent, withCtx, createTextVNode, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderComponent } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  const _component_RouteLink = resolveComponent("RouteLink");
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="数据挖掘概述" tabindex="-1"><a class="header-anchor" href="#数据挖掘概述"><span>数据挖掘概述</span></a></h1><h2 id="为什么学习数据挖掘" tabindex="-1"><a class="header-anchor" href="#为什么学习数据挖掘"><span>为什么学习数据挖掘？</span></a></h2><p>在当今数据爆炸的时代，数据挖掘技术已成为从海量信息中提取价值的关键能力。无论是商业决策、科学研究还是日常生活，数据挖掘都在发挥着越来越重要的作用。本网站旨在帮助你：</p><ul><li>系统掌握数据挖掘的核心概念和方法</li><li>通过实践案例加深对算法的理解</li><li>培养解决实际问题的数据分析能力</li><li>为进一步学习机器学习和人工智能打下基础</li></ul><h2 id="如何使用本网站" tabindex="-1"><a class="header-anchor" href="#如何使用本网站"><span>如何使用本网站？</span></a></h2><ol><li>从`);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/overview/" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`数据挖掘概述`);
      } else {
        return [
          createTextVNode("数据挖掘概述")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`开始，了解基本概念</li><li>参考`);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/learning-path/" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`学习路径图`);
      } else {
        return [
          createTextVNode("学习路径图")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`规划你的学习过程</li><li>深入学习四大`);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/core/preprocessing/data-presentation.html" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`核心知识`);
      } else {
        return [
          createTextVNode("核心知识")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`模块</li><li>通过`);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/projects/" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`实践项目`);
      } else {
        return [
          createTextVNode("实践项目")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`巩固所学知识</li><li>利用`);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/resources/" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`学习资源`);
      } else {
        return [
          createTextVNode("学习资源")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`进一步拓展视野</li></ol><div class="custom-action-buttons"><a href="/learning-path/" class="custom-button primary">查看学习路径</a><a href="/projects/" class="custom-button secondary">开始实践项目</a></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/overview/index.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const index_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "index.html.vue"]]);
const data = JSON.parse('{"path":"/overview/","title":"数据挖掘概述","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"为什么学习数据挖掘？","slug":"为什么学习数据挖掘","link":"#为什么学习数据挖掘","children":[]},{"level":2,"title":"如何使用本网站？","slug":"如何使用本网站","link":"#如何使用本网站","children":[]}],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"overview/README.md"}');
export {
  index_html as comp,
  data
};
