import{_ as l,e,o as s,f as t,F as m,l as d,k as _,x as g,t as r}from"./app-BgnGfOJp.js";const v={name:"TeamMembers",data(){return{members:[{name:"张鹏宇 Тимофей",avatar:"/images/icons/zhang.jpg",color:"#4285F4"},{name:"刘祺泉 Светослав",avatar:"/images/icons/liu.jpg",color:"#EA4335"},{name:"高力柯 Всеволод",avatar:"/images/icons/gao.jpg",color:"#FBBC05"},{name:"葛颂 Игнат",avatar:"/images/icons/葛颂.jpg",color:"#34A853"},{name:"王琦 Леонтий",avatar:"/images/icons/王琦.jpg",color:"#8E44AD"}]}},methods:{getInitial(o){return o?o.charAt(0):"?"}}},p={class:"team-container"},u={class:"team-members"},h={key:0,class:"member-avatar-img"},f=["src","alt"],k={class:"member-initial"},y={class:"member-name"},b={key:2,class:"member-role"};function j(o,B,F,x,n,c){return s(),e("div",p,[t("div",u,[(s(!0),e(m,null,d(n.members,(a,i)=>(s(),e("div",{class:"member-card",key:i},[a.avatar?(s(),e("div",h,[t("img",{src:a.avatar,alt:a.name},null,8,f)])):(s(),e("div",{key:1,class:"member-avatar",style:g({backgroundColor:a.color||"#4285F4"})},[t("span",k,r(c.getInitial(a.name)),1)],4)),t("div",y,r(a.name),1),a.role?(s(),e("div",b,r(a.role),1)):_("",!0)]))),128))])])}const C=l(v,[["render",j],["__scopeId","data-v-fe6d5567"],["__file","TeamMembers.vue"]]);export{C as default};
