<script setup lang="ts">
import {ref} from 'vue'
import Aside from './side/leftside.vue'
import Main1 from '@/views/home/index.vue'
import Main2 from '@/views/person/index.vue'

let clicked = ref(true);
const handleClick = (val: boolean) =>{
  clicked.value = val;
}

</script>


<template>
  <el-container class="home-window">
    <el-aside width="200px" class="home-aside">
      <Aside @click="handleClick" ></Aside>
    </el-aside>
    <el-container class="home-main">
      <el-header class="home-title">
        <el-avatar
        :size="48"
        class="mr-3"
        src="https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png"
        />
      </el-header>
      <el-main class="home-operation">
        <Main1 v-if="clicked"/>
        <Main2 v-else />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.home-window{
  width: 100%;
  height: 100vh;
}

.home-aside{
  box-shadow: 1px 3px 3px black; 
}
.home-title{
height: 10vh;
width: 100%;
display: flex;
justify-content: flex-end;
align-items: center;
box-shadow: 0 1px 3px black;
}

.home-operation{
height: 90vh;
width: 100%;
display: flex;
justify-content:center;
align-items: center;
}
</style>