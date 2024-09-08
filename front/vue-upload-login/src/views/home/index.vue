<script setup lang="ts">
import left from './upload/index.vue'
import right from './show/index.vue'
</script>

<template>
  <div class="main">
    <el-container class="main">
      <el-aside class="left-main"><left/></el-aside>
      <el-main class="right-main"><right/></el-main>
    </el-container>
  </div>
</template>

<style scoped>
.left-main{
  height: 500px;
  width: 500px;
  box-shadow: 1px 1px 3px black;
  background-color: aliceblue;
  Border-radius:20px;
}
.right-main{
  height: 500px;
  width: 500px;
  box-shadow: 1px 1px 3px black;
  border-radius: 20px;
  --el-main-padding:0;
}
.main{
  display: flex;
  justify-content:space-between;
  align-items: center;
}
</style>
