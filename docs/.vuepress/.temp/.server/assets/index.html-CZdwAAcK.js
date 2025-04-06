import { resolveComponent, withCtx, createVNode, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderComponent } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  const _component_ClientOnly = resolveComponent("ClientOnly");
  const _component_ppt_resources = resolveComponent("ppt-resources");
  const _component_doc_resources = resolveComponent("doc-resources");
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="课程资源中心" tabindex="-1"><a class="header-anchor" href="#课程资源中心"><span>课程资源中心</span></a></h1><h2 id="教学课件" tabindex="-1"><a class="header-anchor" href="#教学课件"><span>教学课件</span></a></h2><div class="hint-container tip"><p class="hint-container-title">课件说明</p><p>课件按照课程进度组织，每个主题都包含理论基础和实践案例。建议同学们先观看课件，再进行实践练习。</p></div>`);
  _push(ssrRenderComponent(_component_ClientOnly, null, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(ssrRenderComponent(_component_ppt_resources, null, null, _parent2, _scopeId));
      } else {
        return [
          createVNode(_component_ppt_resources)
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`<h2 id="实践作业" tabindex="-1"><a class="header-anchor" href="#实践作业"><span>实践作业</span></a></h2><div class="hint-container warning"><p class="hint-container-title">作业说明</p><ul><li>所有实践作业都基于 Python 实现</li><li>建议使用 Jupyter Notebook 完成</li><li>每个作业都包含数据集和评分标准</li><li>鼓励同学们互相讨论，但禁止抄袭</li></ul></div>`);
  _push(ssrRenderComponent(_component_ClientOnly, null, {
    default: withCtx((_, _push2, _parent2, _scopeId) => {
      if (_push2) {
        _push2(ssrRenderComponent(_component_doc_resources, null, null, _parent2, _scopeId));
      } else {
        return [
          createVNode(_component_doc_resources)
        ];
      }
    }),
    _: 1
  }, _parent));
  _push(`<h2 id="补充资料" tabindex="-1"><a class="header-anchor" href="#补充资料"><span>补充资料</span></a></h2><h3 id="推荐工具" tabindex="-1"><a class="header-anchor" href="#推荐工具"><span>推荐工具</span></a></h3><ul><li>Python 3.8+</li><li>Jupyter Notebook/Lab</li><li>scikit-learn</li><li>pandas &amp; numpy</li><li>matplotlib &amp; seaborn</li></ul><h3 id="在线资源" tabindex="-1"><a class="header-anchor" href="#在线资源"><span>在线资源</span></a></h3><ul><li><a href="https://scikit-learn.org/" target="_blank" rel="noopener noreferrer">Scikit-learn 官方文档</a></li><li><a href="https://www.kaggle.com/" target="_blank" rel="noopener noreferrer">Kaggle 数据挖掘实战</a></li><li><a href="https://archive.ics.uci.edu/ml/index.php" target="_blank" rel="noopener noreferrer">UCI 机器学习数据集</a></li></ul><h3 id="参考书籍" tabindex="-1"><a class="header-anchor" href="#参考书籍"><span>参考书籍</span></a></h3><ol><li>《数据挖掘：概念与技术》</li><li>《Python数据科学手册》</li><li>《机器学习实战》</li></ol></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/resources/index.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const index_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "index.html.vue"]]);
const data = JSON.parse('{"path":"/resources/","title":"课程资源","lang":"zh-CN","frontmatter":{"title":"课程资源"},"headers":[{"level":2,"title":"教学课件","slug":"教学课件","link":"#教学课件","children":[]},{"level":2,"title":"实践作业","slug":"实践作业","link":"#实践作业","children":[]},{"level":2,"title":"补充资料","slug":"补充资料","link":"#补充资料","children":[{"level":3,"title":"推荐工具","slug":"推荐工具","link":"#推荐工具","children":[]},{"level":3,"title":"在线资源","slug":"在线资源","link":"#在线资源","children":[]},{"level":3,"title":"参考书籍","slug":"参考书籍","link":"#参考书籍","children":[]}]}],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"resources/README.md"}');
export {
  index_html as comp,
  data
};
