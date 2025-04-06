import { resolveComponent, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderComponent } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  const _component_learning_path_visualization = resolveComponent("learning-path-visualization");
  const _component_did_you_know = resolveComponent("did-you-know");
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="学习路径" tabindex="-1"><a class="header-anchor" href="#学习路径"><span>学习路径</span></a></h1><div class="path-intro"><div class="path-intro__content"><h2>你的数据挖掘学习之旅</h2><p>无论你是初学者还是希望提升技能的专业人士，我们都为你准备了清晰的学习路径。按照以下路径学习，你将逐步掌握数据挖掘的核心概念和实践技能。</p><div class="path-intro__features"><div class="path-intro__feature"><div class="path-intro__feature-icon">📚</div><div class="path-intro__feature-text">系统化学习内容</div></div><div class="path-intro__feature"><div class="path-intro__feature-icon">🔍</div><div class="path-intro__feature-text">循序渐进的难度</div></div><div class="path-intro__feature"><div class="path-intro__feature-icon">💻</div><div class="path-intro__feature-text">实践项目巩固</div></div></div></div></div><h2 id="可视化学习路径" tabindex="-1"><a class="header-anchor" href="#可视化学习路径"><span>可视化学习路径</span></a></h2><p>跟随下面的学习路径，逐步掌握数据挖掘的核心知识和技能：</p>`);
  _push(ssrRenderComponent(_component_learning_path_visualization, null, null, _parent));
  _push(`<h2 id="学习建议" tabindex="-1"><a class="header-anchor" href="#学习建议"><span>学习建议</span></a></h2><ol><li><strong>循序渐进</strong>：按照推荐的学习顺序进行学习，每个主题都建立在前一个主题的基础上</li><li><strong>理论结合实践</strong>：学习每个概念后，尝试通过相关的实践项目巩固所学知识</li><li><strong>定期复习</strong>：数据挖掘涉及多个概念和技术，定期回顾已学内容有助于加深理解</li><li><strong>参与讨论</strong>：遇到问题时，可以在社区中提问或与其他学习者交流</li></ol><h2 id="学习资源推荐" tabindex="-1"><a class="header-anchor" href="#学习资源推荐"><span>学习资源推荐</span></a></h2><div class="resource-grid"><div class="resource-card"><div class="resource-card__header"><span class="resource-card__icon">📖</span><h3>初学者指南</h3></div><p>适合零基础学习者的入门资料和学习路径</p><a href="/learning-path/beginner.html" class="resource-card__link">查看详情</a></div><div class="resource-card"><div class="resource-card__header"><span class="resource-card__icon">🚀</span><h3>进阶学习</h3></div><p>深入学习高级算法和技术，提升数据挖掘能力</p><a href="/learning-path/advanced.html" class="resource-card__link">查看详情</a></div><div class="resource-card"><div class="resource-card__header"><span class="resource-card__icon">💼</span><h3>实践应用</h3></div><p>将数据挖掘技术应用到实际业务问题中</p><a href="/learning-path/practical.html" class="resource-card__link">查看详情</a></div></div><div class="did-you-know-container">`);
  _push(ssrRenderComponent(_component_did_you_know, { category: "general" }, null, _parent));
  _push(`</div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/learning-path/index.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const index_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "index.html.vue"]]);
const data = JSON.parse('{"path":"/learning-path/","title":"学习路径","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"可视化学习路径","slug":"可视化学习路径","link":"#可视化学习路径","children":[]},{"level":2,"title":"学习建议","slug":"学习建议","link":"#学习建议","children":[]},{"level":2,"title":"学习资源推荐","slug":"学习资源推荐","link":"#学习资源推荐","children":[]}],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"learning-path/README.md"}');
export {
  index_html as comp,
  data
};
