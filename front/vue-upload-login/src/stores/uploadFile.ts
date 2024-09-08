import { ref, computed, reactive } from 'vue'
import { defineStore } from 'pinia'

export const useUploadFileStore = defineStore('uploadfile', {
  state: () =>({
    file:"",
    source_text:ref("未上传文件"),
    image:ref("https://fuss10.elemecdn.com/a/3f/3302e58f9a181d2509f3dc0fa68b0jpeg.jpeg"),
    reply_text:ref<Array<string>>(["未上传评论","未上传评论","未上传评论","..."]),
    isShow:false,
  }),
  getters:{
    getFile() : string{
      return this.file
    },
    getSourceText() : string{
      return this.source_text
    },
    getReplyText() : Array<string>{
      return this.reply_text
    },
    getImage() : string{
      return this.image
    },
    getIsShow() : boolean{
      return this.isShow
    }
  },
  actions: {
    setfile(uploadfile:string){
      this.file = uploadfile;
    },
    setSourceText(source_text:string){
      if (source_text.length == 0){
        this.source_text = "这是一段默认文本";  
      }else{
        this.source_text = source_text;
      }
    },
    setImage(image:string){
      if (image.length == 0){
        this.image = "https://fuss10.elemecdn.com/a/3f/3302e58f9a181d2509f3dc0fa68b0jpeg.jpeg"
      }else{
        this.image = image;
      }
    },
    setReplyText(reply_text:Array<string>){
      if (reply_text.length == 0){
        this.reply_text = ['目前没有评论','目前没有评论','目前没有评论','...']
      }else{
        this.reply_text = reply_text;
      }
    },
    setIsShow(){
      this.isShow=true;
    },
  }
})
