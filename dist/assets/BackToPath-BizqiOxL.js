import{_ as o,e as s,k as c,o as i,f as n,h,t as l}from"./app-BgnGfOJp.js";const u={name:"BackToPath",data(){return{showBackButton:!1}},computed:{buttonText(){return this.$lang==="en-US"?"Back to Learning Path":"返回学习路径"},pathPrefix(){return this.$lang==="en-US"?"/en":""}},mounted(){const e=document.referrer,t=window.location.host;e.includes(t)&&(e.includes("/learning-path/")||e.includes("/en/learning-path/"))&&(this.showBackButton=!0),localStorage.getItem("fromLearningPath")==="true"&&(this.showBackButton=!0,localStorage.removeItem("fromLearningPath"))},created(){this.$nextTick(()=>{document.querySelectorAll('a[href*="/learning-path/"]').forEach(t=>{t.addEventListener("click",()=>{sessionStorage.setItem("fromLearningPath","true")})})})}},d={key:0,class:"back-to-path"},f=["href"];function k(e,t,p,_,r,a){return r.showBackButton?(i(),s("div",d,[n("a",{href:a.pathPrefix+"/learning-path/",class:"back-to-path__button"},[t[0]||(t[0]=n("span",{class:"back-to-path__icon"},"←",-1)),h(" "+l(a.buttonText),1)],8,f)])):c("",!0)}const g=o(u,[["render",k],["__scopeId","data-v-903e279e"],["__file","BackToPath.vue"]]);export{g as default};
