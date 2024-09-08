import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path:'/',
      redirect:'/login'  //路由重定向
    },
    {
      path:'/home',   // 主页
      name:'home',
      component:()=>import('@/layout/index.vue'),
    },
    {
      path:'/login',   // 登录页
      name:'login',
      component:()=>import('@/views/login/index.vue'),
    },
  ]
})

router.beforeEach( (to,from,next) =>{
  const token = localStorage.getItem('user-token');//查看本地存储上是否有user-token对象
  if( token || to.path === '/login'){//短路逻辑，有就可以继续执行，没有就跳转到登录页面
      //console.log(token);
      next();
  }else{
   next( '/login' );//跳转登录页面
   alert("请先登录系统");
  }
})

export default router
