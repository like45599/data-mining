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
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="learning-path" tabindex="-1"><a class="header-anchor" href="#learning-path"><span>Learning Path</span></a></h1><div class="path-intro"><div class="path-intro__content"><h2>Your Data Mining Learning Journey</h2><p>Whether you are a beginner or a professional looking to enhance your skills, we have prepared a clear learning path for you. By following this path, you will gradually master the core concepts and practical skills of data mining.</p><div class="path-intro__features"><div class="path-intro__feature"><div class="path-intro__feature-icon">📚</div><div class="path-intro__feature-text">Systematic Learning Content</div></div><div class="path-intro__feature"><div class="path-intro__feature-icon">🔍</div><div class="path-intro__feature-text">Gradual Difficulty Progression</div></div><div class="path-intro__feature"><div class="path-intro__feature-icon">💻</div><div class="path-intro__feature-text">Hands-on Projects for Reinforcement</div></div></div></div></div><h2 id="visual-learning-path" tabindex="-1"><a class="header-anchor" href="#visual-learning-path"><span>Visual Learning Path</span></a></h2><p>Follow the learning path below to gradually master the core knowledge and skills of data mining:</p>`);
  _push(ssrRenderComponent(_component_learning_path_visualization, null, null, _parent));
  _push(`<h2 id="learning-recommendations" tabindex="-1"><a class="header-anchor" href="#learning-recommendations"><span>Learning Recommendations</span></a></h2><ol><li><strong>Gradual Progression</strong>: Follow the recommended learning order, where each topic builds on the previous one.</li><li><strong>Theory Combined with Practice</strong>: After learning each concept, try to reinforce your knowledge by working on related practical projects.</li><li><strong>Regular Review</strong>: Data mining involves multiple concepts and techniques. Regularly reviewing what you have learned will help deepen your understanding.</li><li><strong>Participate in Discussions</strong>: If you encounter issues, you can ask questions in the community or interact with other learners.</li></ol><h2 id="recommended-learning-resources" tabindex="-1"><a class="header-anchor" href="#recommended-learning-resources"><span>Recommended Learning Resources</span></a></h2><div class="resource-grid"><div class="resource-card"><div class="resource-card__header"><span class="resource-card__icon">📖</span><h3>Beginner&#39;s Guide</h3></div><p>Introductory materials and learning paths for absolute beginners</p><a href="/en/learning-path/beginner.html" class="resource-card__link">Learn More</a></div><div class="resource-card"><div class="resource-card__header"><span class="resource-card__icon">🚀</span><h3>Advanced Learning</h3></div><p>Deep dive into advanced algorithms and techniques to enhance your data mining capabilities</p><a href="/en/learning-path/advanced.html" class="resource-card__link">Learn More</a></div><div class="resource-card"><div class="resource-card__header"><span class="resource-card__icon">💼</span><h3>Practical Applications</h3></div><p>Apply data mining techniques to real-world business problems</p><a href="/en/learning-path/practical.html" class="resource-card__link">Learn More</a></div></div><div class="did-you-know-container">`);
  _push(ssrRenderComponent(_component_did_you_know, { category: "general" }, null, _parent));
  _push(`</div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/en/learning-path/index.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const index_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "index.html.vue"]]);
const data = JSON.parse('{"path":"/en/learning-path/","title":"Learning Path","lang":"en-US","frontmatter":{},"headers":[{"level":2,"title":"Visual Learning Path","slug":"visual-learning-path","link":"#visual-learning-path","children":[]},{"level":2,"title":"Learning Recommendations","slug":"learning-recommendations","link":"#learning-recommendations","children":[]},{"level":2,"title":"Recommended Learning Resources","slug":"recommended-learning-resources","link":"#recommended-learning-resources","children":[]}],"git":{"updatedTime":1742831857000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":2,"url":"https://github.com/like45599"}],"changelog":[{"hash":"2bc457cfaf02a69e1673760e9106a75f7cced3da","time":1742831857000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"优化跳转地址+更新网站icon"},{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"en/learning-path/README.md"}');
export {
  index_html as comp,
  data
};
