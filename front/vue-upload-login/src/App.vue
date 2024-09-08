<script setup lang="ts">
import { RouterView } from 'vue-router'
</script>

<template>
    <div class="home">
        <RouterView />
    </div>
</template>

<style scoped>
.home{
  height: 100vh;
  width: 100%;
  background-color: aliceblue;
  position: fixed;
  top: 0;
  left: 0;
  background:url(@/assets/image/3.png) !important;
}
</style>
