import { resolveComponent, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderComponent } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  const _component_TeamMembers = resolveComponent("TeamMembers");
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="about-us" tabindex="-1"><a class="header-anchor" href="#about-us"><span>About Us</span></a></h1><p>This data mining tutorial was developed by the following members:</p>`);
  _push(ssrRenderComponent(_component_TeamMembers, null, null, _parent));
  _push(`<h2 id="contact-us" tabindex="-1"><a class="header-anchor" href="#contact-us"><span>Contact Us</span></a></h2><p>If you have any questions or suggestions, please contact us through the following ways:</p><ul><li>Email: 1271383559@qq.com</li><li>GitHub: <a href="https://github.com/like45599" target="_blank" rel="noopener noreferrer">github.com/like45599</a></li></ul></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/en/about/index.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const index_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "index.html.vue"]]);
const data = JSON.parse('{"path":"/en/about/","title":"About Us","lang":"en-US","frontmatter":{},"headers":[{"level":2,"title":"Contact Us","slug":"contact-us","link":"#contact-us","children":[]}],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"en/about/README.md"}');
export {
  index_html as comp,
  data
};
