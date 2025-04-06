import { defineAsyncComponent } from 'vue'

export default {
  enhance: ({ app }) => {    
      app.component("BackToPath", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/BackToPath.vue")))
    
      app.component("CaseStudies", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/CaseStudies.vue")))
    
      app.component("chart-example", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/chart-example.vue")))
    
      app.component("CrispDmModel", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/CrispDmModel.vue")))
    
      app.component("data-representation", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/data-representation.vue")))
    
      app.component("did-you-know", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/did-you-know.vue")))
    
      app.component("DisciplineMap", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/DisciplineMap.vue")))
    
      app.component("doc-resources", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/doc-resources.vue")))
    
      app.component("home-features", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/home-features.vue")))
    
      app.component("input-form", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/input-form.vue")))
    
      app.component("LanguageSwitcher", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/LanguageSwitcher.vue")))
    
      app.component("learning-path-visualization", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/learning-path-visualization.vue")))
    
      app.component("MathJaxLoader", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/MathJaxLoader.vue")))
    
      app.component("ppt-resources", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/ppt-resources.vue")))
    
      app.component("practice-notebook", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/practice-notebook.vue")))
    
      app.component("resource-cards", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/resource-cards.vue")))
    
      app.component("TeamMembers", defineAsyncComponent(() => import("D:/GAOLIKE/Java/项目/大学社团管理系统/RuoYi-Vue3/data-mining/docs/.vuepress/components/TeamMembers.vue")))
  },
}
