import { ssrRenderAttrs } from "vue/server-renderer";
import { useSSRContext } from "vue";
import { _ as _export_sfc } from "../app.C2n1Acj7.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="聚类分析" tabindex="-1"><a class="header-anchor" href="#聚类分析"><span>聚类分析</span></a></h1><h2 id="k-means算法" tabindex="-1"><a class="header-anchor" href="#k-means算法"><span>K-Means算法</span></a></h2><ol><li>算法原理</li><li>参数选择</li><li>优缺点分析</li></ol><h2 id="评估指标" tabindex="-1"><a class="header-anchor" href="#评估指标"><span>评估指标</span></a></h2><ol><li>轮廓系数</li><li>肘部法则</li><li>簇间距离</li></ol><h2 id="应用场景" tabindex="-1"><a class="header-anchor" href="#应用场景"><span>应用场景</span></a></h2><ol><li>客户分群</li><li>图像分割</li><li>文档聚类</li></ol><h2 id="实践案例" tabindex="-1"><a class="header-anchor" href="#实践案例"><span>实践案例</span></a></h2><p>[待补充具体案例代码和数据]</p></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/guide/clustering.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const clustering_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "clustering.html.vue"]]);
const data = JSON.parse('{"path":"/guide/clustering.html","title":"聚类分析","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"K-Means算法","slug":"k-means算法","link":"#k-means算法","children":[]},{"level":2,"title":"评估指标","slug":"评估指标","link":"#评估指标","children":[]},{"level":2,"title":"应用场景","slug":"应用场景","link":"#应用场景","children":[]},{"level":2,"title":"实践案例","slug":"实践案例","link":"#实践案例","children":[]}],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"guide/clustering.md"}');
export {
  clustering_html as comp,
  data
};
