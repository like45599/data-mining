import { ssrRenderAttrs } from "vue/server-renderer";
import { useSSRContext } from "vue";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs) {
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="预测与回归分析" tabindex="-1"><a class="header-anchor" href="#预测与回归分析"><span>预测与回归分析</span></a></h1><h2 id="回归类型" tabindex="-1"><a class="header-anchor" href="#回归类型"><span>回归类型</span></a></h2><ol><li>线性回归</li><li>多项式回归</li><li>逻辑回归</li></ol><h2 id="模型评估" tabindex="-1"><a class="header-anchor" href="#模型评估"><span>模型评估</span></a></h2><ol><li>MSE</li><li>RMSE</li><li>R²值</li><li>调整R²</li></ol><h2 id="特征工程" tabindex="-1"><a class="header-anchor" href="#特征工程"><span>特征工程</span></a></h2><ol><li>特征选择</li><li>特征缩放</li><li>特征交叉</li></ol><h2 id="实践案例" tabindex="-1"><a class="header-anchor" href="#实践案例"><span>实践案例</span></a></h2><p>[待补充具体案例代码和数据]</p></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/.temp/pages/guide/prediction.html.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const prediction_html = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__file", "prediction.html.vue"]]);
const data = JSON.parse('{"path":"/guide/prediction.html","title":"预测与回归分析","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"回归类型","slug":"回归类型","link":"#回归类型","children":[]},{"level":2,"title":"模型评估","slug":"模型评估","link":"#模型评估","children":[]},{"level":2,"title":"特征工程","slug":"特征工程","link":"#特征工程","children":[]},{"level":2,"title":"实践案例","slug":"实践案例","link":"#实践案例","children":[]}],"git":{"updatedTime":1742460681000,"contributors":[{"name":"like45599","username":"like45599","email":"131803211+like45599@users.noreply.github.com","commits":1,"url":"https://github.com/like45599"}],"changelog":[{"hash":"61a31e55d1325755fa12a32e909ee09c3ac0a20f","time":1742460681000,"email":"131803211+like45599@users.noreply.github.com","author":"Yun Feng","message":"数据挖掘指南v1.0"}]},"filePathRelative":"guide/prediction.md"}');
export {
  prediction_html as comp,
  data
};
