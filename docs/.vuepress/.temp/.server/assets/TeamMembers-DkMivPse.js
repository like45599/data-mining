import { mergeProps, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderList, ssrRenderAttr, ssrRenderStyle, ssrInterpolate } from "vue/server-renderer";
import { _ as _export_sfc } from "../app.DfozzGWd.mjs";
import "@vuepress/shared";
import "vue-router";
import "@vueuse/core";
import "@vue/devtools-api";
const _sfc_main = {
  name: "TeamMembers",
  data() {
    return {
      members: [
        {
          name: "张鹏宇 Тимофей",
          avatar: "/images/icons/zhang.jpg",
          color: "#4285F4"
        },
        {
          name: "刘祺泉 Светослав",
          avatar: "/images/icons/liu.jpg",
          color: "#EA4335"
        },
        {
          name: "高力柯 Всеволод",
          avatar: "/images/icons/gao.jpg",
          color: "#FBBC05"
        },
        {
          name: "葛颂 Игнат",
          avatar: "/images/icons/葛颂.jpg",
          color: "#34A853"
        },
        {
          name: "王琦 Леонтий",
          avatar: "/images/icons/王琦.jpg",
          color: "#8E44AD"
        }
      ]
    };
  },
  methods: {
    getInitial(name) {
      return name ? name.charAt(0) : "?";
    }
  }
};
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(mergeProps({ class: "team-container" }, _attrs))} data-v-fe6d5567><div class="team-members" data-v-fe6d5567><!--[-->`);
  ssrRenderList($data.members, (member, index) => {
    _push(`<div class="member-card" data-v-fe6d5567>`);
    if (member.avatar) {
      _push(`<div class="member-avatar-img" data-v-fe6d5567><img${ssrRenderAttr("src", member.avatar)}${ssrRenderAttr("alt", member.name)} data-v-fe6d5567></div>`);
    } else {
      _push(`<div class="member-avatar" style="${ssrRenderStyle({ backgroundColor: member.color || "#4285F4" })}" data-v-fe6d5567><span class="member-initial" data-v-fe6d5567>${ssrInterpolate($options.getInitial(member.name))}</span></div>`);
    }
    _push(`<div class="member-name" data-v-fe6d5567>${ssrInterpolate(member.name)}</div>`);
    if (member.role) {
      _push(`<div class="member-role" data-v-fe6d5567>${ssrInterpolate(member.role)}</div>`);
    } else {
      _push(`<!---->`);
    }
    _push(`</div>`);
  });
  _push(`<!--]--></div></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add(".vuepress/components/TeamMembers.vue");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const TeamMembers = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender], ["__scopeId", "data-v-fe6d5567"], ["__file", "TeamMembers.vue"]]);
export {
  TeamMembers as default
};
