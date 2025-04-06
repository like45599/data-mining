import { resolveComponent, withCtx, createTextVNode, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderComponent } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  const _component_RouteLink = resolveComponent("RouteLink");
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="data-mining-overview" tabindex="-1"><a class="header-anchor" href="#data-mining-overview"><span>Data Mining Overview</span></a></h1><h2 id="why-learn-data-mining" tabindex="-1"><a class="header-anchor" href="#why-learn-data-mining"><span>Why Learn Data Mining?</span></a></h2><p>In today&#39;s age of data explosion, data mining techniques have become a key capability for extracting value from vast amounts of information. Whether it&#39;s business decision-making, scientific research, or daily life, data mining plays an increasingly important role. This website aims to help you:</p><ul><li>Systematically master the core concepts and methods of data mining</li><li>Deepen your understanding of algorithms through practical case studies</li><li>Develop data analysis skills to solve real-world problems</li><li>Lay the foundation for further learning in machine learning and artificial intelligence</li></ul><h2 id="how-to-use-this-website" tabindex="-1"><a class="header-anchor" href="#how-to-use-this-website"><span>How to Use This Website?</span></a></h2><ol><li>Start with the `);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/en/overview/" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`Data Mining Overview`);
      } else {
        return [
          createTextVNode("Data Mining Overview")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(` to understand the basic concepts</li><li>Refer to the `);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/en/learning-path/" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`Learning Path`);
      } else {
        return [
          createTextVNode("Learning Path")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(` to plan your learning process</li><li>Dive into the four `);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/en/core/preprocessing/data-presentation.html" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`Core Knowledge`);
      } else {
        return [
          createTextVNode("Core Knowledge")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(` modules</li><li>Consolidate what you&#39;ve learned through `);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/en/projects/" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`Practical Projects`);
      } else {
        return [
          createTextVNode("Practical Projects")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`</li><li>Utilize `);
  _push(ssrRenderComponent(_component_RouteLink, { to: "/en/resources/" }, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(`Learning Resources`);
      } else {
        return [
          createTextVNode("Learning Resources")
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(` to further expand your knowledge</li></ol><div class="custom-action-buttons"><a href="/en/learning-path/" class="custom-button primary">View Learning Path</a><a href="/en/projects/" class="custom-button secondary">Start Practical Projects</a></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/en/overview/index.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const index_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "index.html.vue"]]);
const data = JSON.parse('{"path":"/en/overview/","title":"Data Mining Overview","lang":"en-US","frontmatter":{},"headers":[{"level":2,"title":"Why Learn Data Mining?","slug":"why-learn-data-mining","link":"#why-learn-data-mining","children":[]},{"level":2,"title":"How to Use This Website?","slug":"how-to-use-this-website","link":"#how-to-use-this-website","children":[]}],"git":{"updatedTime":1742831857000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":2,"url":"https://github.com/like45599"}],"changelog":[{"hash":"2bc457cfaf02a69e1673760e9106a75f7cced3da","time":1742831857000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"优化跳转地址+更新网站icon"},{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"en/overview/README.md"}');
export {
  index_html as comp,
  data
};
