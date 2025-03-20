import DisciplineMap from './components/DisciplineMap.vue'
import CrispDmModel from './components/CrispDmModel.vue'
import CaseStudies from './components/CaseStudies.vue'
import TeamMembers from './components/TeamMembers.vue'

export default ({
  Vue, // VuePress 正在使用的 Vue 构造函数
}) => {
  // 注册组件
  Vue.component('DisciplineMap', DisciplineMap)
  Vue.component('CrispDmModel', CrispDmModel)
  Vue.component('CaseStudies', CaseStudies)
  Vue.component('TeamMembers', TeamMembers)
} 