<script setup lang="ts">
import { ref } from 'vue';
import uploadJson from './Json.vue'
import form from './Json.vue'
import uploadTxt from './Txt.vue'

let Jsonshow = ref(true)
let txtshow = ref(false)

const changeTxt = ()=>{
  Jsonshow.value = false;
  txtshow.value = true;
}
const changeJson = ()=>{
  Jsonshow.value = true;
  txtshow.value = false;
}

</script>


<template>
<div class="mainleft">
  <el-container class="cl-minleft">
    <el-header class="header">
      <el-button type="primary" round @click="changeTxt">上传文本数据</el-button>
      <el-button type="primary" round @click="changeJson">上传json数据</el-button>
    </el-header>
    <el-main class="main">
      <uploadJson v-if="Jsonshow"/>
      <uploadTxt v-if="txtshow"/>
    </el-main>
  </el-container>
</div>
</template>

<style scoped>
.header{
  display: flex;
  justify-content: center;
  align-items: center;
}
.main{
  display: flex;
  justify-content: center;
  align-items: center;
}
.mainleft{
  height: 100%;
}
.cl-minleft{
  height: 100%;
}
</style>