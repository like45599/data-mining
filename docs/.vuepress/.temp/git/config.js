import { GitContributors } from "D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/node_modules/@vuepress/plugin-git/lib/client/components/GitContributors.js";
import { GitChangelog } from "D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/node_modules/@vuepress/plugin-git/lib/client/components/GitChangelog.js";

export default {
  enhance: ({ app }) => {
    app.component("GitContributors", GitContributors);
    app.component("GitChangelog", GitChangelog);
  },
};
