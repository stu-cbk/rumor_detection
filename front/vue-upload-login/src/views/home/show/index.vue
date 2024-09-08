<script setup lang="ts">
import {ref,watch} from 'vue'
import vueDanmaku from 'vue3-danmaku'
import { useUploadFileStore } from '@/stores/uploadFile'
import axios from '@/http'
import shishi from '@/assets/image/shishi.png'
import yaoyan from '@/assets/image/yaoyan.png'
import Json from '../upload/Json.vue'

const uploadStore = useUploadFileStore()

const file = ref(uploadStore.getFile)
const source = ref(uploadStore.getSourceText)
const danmus = ref(uploadStore.getReplyText)
const speeds = 100
const url = ref(uploadStore.getImage)
const srcList = ref([uploadStore.getImage,])
const show =  ref(uploadStore.getIsShow)

const resurls = [shishi,yaoyan]

let typeurl = ref(0)
let isShow = ref(false)
let loading = ref(false)
let ratetext = ref("")

const svg = `
        <path class="path" d="
          M 30 15
          L 28 17
          M 25.61 25.61
          A 15 15, 0, 0, 1, 15 30
          A 15 15, 0, 1, 1, 27.99 7.5
          L 15 15
        " style="stroke-width: 4px; fill: rgba(0, 0, 0, 0)"/>
        `

// 监听 pinia 是否发生变化
watch(
    () => [uploadStore.file,uploadStore.source_text, uploadStore.reply_text,uploadStore.image,uploadStore.isShow],
    ([new_file,new_source_text, new_reply_text,new_image,new_isShow], 
    [old_file,old_source_text, old_reply_text,old_image,old_isShow]) => {
      file.value = new_file as string;
      source.value = new_source_text as string;
      danmus.value = new_reply_text as Array<string>;
      url.value = new_image as string;
      srcList.value = [new_image,] as Array<string>;
      show.value = new_isShow as boolean;
    }
);

const handleSubmit = async() =>{
  // 数据交互 传输json字符串数据
  
  // axios
  loading.value = true;
  let res = await axios.post('v1/upload',file.value,{
    headers:{'Content-Type':'application/json;'}
  });
  loading.value = false;
  isShow.value = true;
  typeurl.value = res.data[0]['isRumor'];
  let rate = parseFloat(res.data[0]['rate']).toFixed(2);
  ratetext.value = rate + "%";
  console.log(ratetext);
}

</script>

<template>
<transition name="el-zoom-in-top">
  <el-card class="card" v-if="show">
    <template #header class="header">
      <el-popover
      v-model:content="source"
      placement="top-start"
      title="源文本"
      :width="200"
      trigger="hover"
      class="popover">
        <template #reference class="popover">
          <el-button type="primary" round class="m-2">源文本</el-button>
        </template>
      </el-popover>
    </template>
      <div class="body"
        v-loading="loading"
        element-loading-text="Loading..."
        :element-loading-spinner="svg"
        element-loading-svg-view-box="-10, -10, 50, 50"
        element-loading-background="rgba(122, 122, 122, 0.8)"
      >
      <el-container class="main">
        <el-header class="header-main">
          <vue-danmaku 
          v-model:danmus="danmus"
          v-model:speeds="speeds"
          extraStyle="color:#23B2B7"
          loop
          class="danmu" 
        ></vue-danmaku>
        </el-header>
        <el-main class="main-main">
          <el-image
            style="width: 100px; height: 100px"
            :src="url"
            :preview-src-list="srcList"
            :zoom-rate="1.2"
            :max-scale="7"
            :min-scale="0.2"
            :initial-index="4"
            fit="cover"
          />
          <el-popover
          v-model:content="ratetext"
          placement="right-start"
          title="预测概率"
          :width="200"
          trigger="hover"
          > 
          <template #reference>
            <el-image
              v-if="isShow"
              style="width: 100px; height: 100px"
              :src="resurls[typeurl]"
              fit="cover">
            </el-image>
          </template>
          </el-popover>
        </el-main>
      </el-container>
      </div>
    <template #footer class="footer">
        <el-button type="success" round class="m-2" @click="handleSubmit">谣言检测</el-button>
    </template>
  </el-card>
</transition>
</template>

<style scoped>
.card{
  height: 100%;
  background-color: aliceblue;
  border:0;
}
.header{
  height: 50px;
}
.body{
  height: 300px;
  display: flex;
}
.footer{
  height: 50px;
}
.m-2{
  display:block;
  margin:0 auto;
}
.main-main{
  height: 200px;
  display: flex;
  justify-content:center;
  align-items: center;
}
.danmu{
  height:100px; 
  width:400px;
}
</style>