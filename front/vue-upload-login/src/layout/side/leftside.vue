<script lang="ts" setup>
import {ref,defineExpose} from 'vue'
import {User,House,Grid,Monitor} from '@element-plus/icons-vue'
import axios from '@/http'
import {userDataStore} from '@/stores/userData'
import { da } from 'element-plus/es/locale/index.mjs';

// 收放表
const handleOpen = (key: string, keyPath: string[]) => {
    console.log(key, keyPath)
}
const handleClose = (key: string, keyPath: string[]) => {
    console.log(key, keyPath)
}

const userStore = userDataStore();

// 初始化谣言库页面
const init = async() => {
  let res1 = await axios.get('v2/getUserData',{
        headers: {
          'Authorization': 'Bearer ' + localStorage.getItem('user-token')
        },
        params: {//请求参数（条件查询）
            "token":localStorage.getItem("user-token")
        }
    });
  
  if (res1.data['status'] == 400) {console.log("初始化失败");return}
  let data1 = res1.data['data'];
  if (data1['username'] != userStore.getUsername)
  {
    userStore.setUsername(data1['username'])
    userStore.setPhone(data1['phone'])
    userStore.setEmail(data1['email'])
  }

  let res2 = await axios.get('v2/getRate',{
    headers: {
          'Authorization': 'Bearer ' + localStorage.getItem('user-token')
        },
  })

  if (res2.data['status'] == 400) {console.log("初始化失败");return}
  let data2 = res2.data['data']
  if (data2[0] != userStore.getNotRumor || data2[1] != userStore.getIsRumor)
  {
    userStore.setNotRumor(data2[0])
    userStore.setIsRumor(data2[1])
  }
  //console.log(userStore.getRumorList)
  let res3 = await axios.get('v2/getRumorList',{
    headers: {
          'Authorization': 'Bearer ' + localStorage.getItem('user-token')
        },
  })

  if (res3.data['status'] == 400) {console.log("初始化失败");return}
  let data3 = res3.data['data']
  //console.log(data3)
  userStore.setRumorList(data3)
  
}

// 切换页面
let clicked = ref(true);
const emits = defineEmits(["click"])
const goToWorkbench = () => {
  clicked.value = true;
  emits("click",clicked.value);
}
const goToRumorLibrary = () => {
  clicked.value = false;
  init();
  emits("click",clicked.value);
}
 
</script>

<template>
    <el-row class="tac">
      <el-col :span="24">
        <h2 class="mb-2">导航栏</h2>
        <el-menu
          default-active="1-1"
          class="el-menu-vertical-demo"
          @open="handleOpen"
          @close="handleClose"
        >
          <el-sub-menu index="1" class="left-icon">
            <template #title>
              <el-icon><House /></el-icon>
              <span>主页</span>
            </template>
              <el-menu-item index="1-1" class="left-icon" @click="goToWorkbench">
                <el-icon><Monitor /></el-icon>
                <span>工作台</span>
              </el-menu-item>
          </el-sub-menu>
          <el-menu-item index="2" class="left-icon" @click="goToRumorLibrary">
            <el-icon><Grid /></el-icon>
            <span>谣言库</span>
          </el-menu-item>
        </el-menu>
      </el-col>
    </el-row>
</template>
  
<style scoped>

.mb-2{
  text-align: center;
  padding: 20px;
  color: aliceblue;
}

.el-menu-vertical-demo{
  opacity: 0.7;
  background-color:azure;
}
</style>
  