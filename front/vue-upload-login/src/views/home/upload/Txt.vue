<script lang="ts" setup>
import { ref,reactive } from 'vue'
import { ElMessage, ElMessageBox, genFileId } from 'element-plus'
import type { UploadInstance,UploadUserFile,UploadProps,UploadRawFile } from 'element-plus'
import { useUploadFileStore } from '@/stores/uploadFile';

const textArea = ref("")
const upload = ref<UploadInstance>()
const fileList = ref<UploadUserFile[]>([])
const reader = new FileReader()
const uploadFileData = useUploadFileStore()

let form = reactive({file:""})

const handleRemove: UploadProps['onRemove'] = (file, uploadFiles) => {
  console.log(file, uploadFiles)
}

const handlePreview: UploadProps['onPreview'] = (uploadFile) => {
  console.log(uploadFile)
}

const handleExceed: UploadProps['onExceed'] = (files, uploadFiles) => {
  upload.value!.clearFiles()
  const file = files[0] as UploadRawFile
  file.uid = genFileId()
  upload.value!.handleStart(file)
}

const uploadFile = (params: any) => {
  let blob = new Blob([params.file])
  reader.readAsDataURL(blob)
  reader.onload = () => {
    let jsondata = JSON.stringify([{
      "source_text":textArea.value,
      "image":reader.result as string,
      "reply_text":[]
    }]);
    form.file = jsondata;
  };
}

const submitUpload = () => {
  upload.value!.clearFiles()
  if (form.file.length > 0){
    console.log(JSON.parse(form.file))
    let data = JSON.parse(form.file)
    uploadFileData.setfile(form.file)
    uploadFileData.setSourceText(data[0]["source_text"])
    uploadFileData.setImage(data[0]["image"])
    uploadFileData.setIsShow()
  }else{
    let text = '这是默认文本';
    if (textArea.value.length > 0)
      {text = textArea.value}
    let jsondata = JSON.stringify([{
      "source_text":text,
      "image":"",
      "reply_text":[]
    }]);
    uploadFileData.setfile(jsondata);
    uploadFileData.setSourceText(text);
    uploadFileData.setIsShow();
  }
}
</script>


<template>
  <el-container class="txt">
    <el-header class="header">
      <el-container class="txt-header">
        <el-header class="header-header">
          <el-input
            v-model="textArea"
            style="width: 500px"
            :rows="3"
            type="textarea"
            placeholder="请输入可疑信息"
          />
        </el-header>
        <el-main class="header-main">
            <el-upload
              v-model:file-list="fileList"
              ref="upload"
              class="upload-txt"
              action=""
              accept="image/png, image/jpeg"
              :http-request="uploadFile"
              :on-preview="handlePreview"
              :on-remove="handleRemove"
              :on-exceed="handleExceed"
              :limit="1"
            >
            <template #trigger class="image-button">
              <el-button type="primary">附图</el-button>
            </template>
            </el-upload>
        </el-main>
      </el-container>
    </el-header>
    <el-main class="main">
        <el-button type="success" round @click="submitUpload">上传文件</el-button>
    </el-main>
  </el-container>
</template>
  
<style scoped>
.header{
  display: flex;
  justify-content: center;
  align-items: center;
  height: 80%;
}
.main{
  display: flex;
  justify-content: center;
  align-items: center;
  height: 20%;
}
.txt{
  height: 100%;
  width: 100%;
}
.txt-header{
  height: 100%;
  width: 100%;
}
.header-header{
  display: flex;
  justify-content: center;
  align-items: center;
  height: 50%;
}
.header-main{
  display: flex;
  justify-content: center;
  align-items: center;
  height: 50%;
}
.upload-txt{
  opacity: 0.8;
  height: 80%;
}

</style>