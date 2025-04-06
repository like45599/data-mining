import { CodeTabs } from "D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/node_modules/@vuepress/plugin-markdown-tab/lib/client/components/CodeTabs.js";
import { Tabs } from "D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/node_modules/@vuepress/plugin-markdown-tab/lib/client/components/Tabs.js";
import "D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/node_modules/@vuepress/plugin-markdown-tab/lib/client/styles/vars.css";

export default {
  enhance: ({ app }) => {
    app.component("CodeTabs", CodeTabs);
    app.component("Tabs", Tabs);
  },
};
