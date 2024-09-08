<script setup lang="ts">
import { UploadFilled } from '@element-plus/icons-vue'
import { ref,reactive } from 'vue'
import { ElMessage, ElMessageBox, genFileId } from 'element-plus'
import type { UploadInstance, UploadProps, UploadRawFile,UploadUserFile} from 'element-plus'
import { useUploadFileStore } from '@/stores/uploadFile';
import router from '@/router';


const fileList = ref<UploadUserFile[]>([])

const upload = ref<UploadInstance>()

const reader = new FileReader()

const form = reactive({file:""})

const uploadFileData = useUploadFileStore()

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

const submitUpload = () => {
  upload.value!.clearFiles()
  if (form.file.length > 0){
    console.log(JSON.parse(form.file))
    let data = JSON.parse(form.file)
    uploadFileData.setfile(form.file)
    uploadFileData.setSourceText(data[0]["source_text"])
    uploadFileData.setReplyText(data[0]["reply_text"])
    uploadFileData.setImage(data[0]["image"])
    uploadFileData.setIsShow()
  }
}

const uploadFile = (params: any) => {
  let blob = new Blob([params.file])
  reader.readAsText(blob,'UTF-8')
  reader.onload = () => {
    form.file = reader.result as string;
  };
}

</script>

<template>
  <el-container class="json">
  <el-header class="header">
    <el-upload
      v-model:file-list="fileList"
      ref="upload"
      class="upload-json"
      drag
      action=""
      accept=".json"
      :http-request="uploadFile"
      :on-preview="handlePreview"
      :on-remove="handleRemove"
      :on-exceed="handleExceed"
      :limit="1"
    >
      <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
      <div class="el-upload__text">
        拖曳文件上传 或者 <em>点击上传</em>
      </div>
      <template #tip>
      <div class="el-upload__tip">
        limit 1 file, new file will cover the old file
      </div>
    </template> 
    </el-upload>
  </el-header>
  <el-main class="main">
      <el-button type="success" round  @click="submitUpload">上传文件</el-button>
  </el-main>
  </el-container>
</template>

<style scoped>
.upload-json{
  opacity: 0.8;
}
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
.json{
  height: 100%;
  width: 100%;
}
.el-upload__tip{
  color: red;
  size: 2px;
}
</style>