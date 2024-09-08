<script setup lang="ts">
import {ref,reactive} from 'vue'
import { sha256 } from 'js-sha256'
import { JSEncrypt } from 'jsencrypt'
import axios from '@/http'
import { useGetDerivedNamespace } from 'element-plus';
import { ElNotification } from 'element-plus'
import {useRouter} from 'vue-router'//导入vue-router

const salt = "0#$@%&4386adgakjg";
const publicKey = "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCugFFJUMStLfalFd\
  kn9R3z8Pvzbyv/6/y07gjuAq6MhECdnZRrfjaIj7PX+NiPSxGPXHpb/olVo2VBHuOTbgXyYC\
  XKAqHB7xATzxCqLW92SjLWQ9pAhxGHlKGy3z0+wgeqOcegceG7qNw4b29SN5tW+gykDeWyQ1tvcye4FnvcRQIDAQAB";

let error = reactive({
  uesrEmpty:false,
  userOversize:false,
  userExisted:false,

  emailEmpty:false,
  emailError:false,

  phoneEmpty:false,
  phoneError:false,

  pwdEmpty:false,
  pwdOversize:false,
  pwdError:false,
})

let isLogin = ref(true)

let form = reactive({
  username:'',
	useremail:'',
  userphone:'',
	userpwd:''
})

const setError = () =>{
  error.uesrEmpty = false;
  error.userOversize = false;
  error.userExisted = false;

  error.emailEmpty = false;
  error.emailError = false;
  
  error.phoneEmpty = false;
  error.phoneError = false;

  error.pwdEmpty = false;
  error.pwdOversize = false;
  error.pwdError = false;
}

const router = useRouter();

const login = async() =>{
  setError();
  if (form.username == '') {error.uesrEmpty = true;return;}
  if (form.userpwd == '') {error.pwdEmpty= true;return;}
  let res1 = await axios.get('v2/judgeNameValid',{
    params: {//请求参数（条件查询）
      username:form.username
    }
  });
  if (res1.data['status'] == 200) {alert("用户名或密码填写错误");return;}
  let hashpwd = sha256(form.userpwd + salt);
  let encryptor = new JSEncrypt();
  encryptor.setPublicKey(publicKey);
  let rsapwd = encryptor.encrypt(hashpwd);
  let jsonData = JSON.stringify({
    "id":"",
    "username":form.username,
    "password":rsapwd as string,
    "phone":"",
    "email":"",
  });
  let res2 = await axios.post('v2/login',jsonData,{
    headers:{'Content-Type':'application/json;'}
  });
  if (res2.data['status'] == 200) {
    localStorage.setItem("user-token",res2.data['data']);
    router.push('/home');
  }
  else {alert("用户名或密码填写错误");}
}
const register = async() =>{
  setError();
  if (form.username == '') {error.uesrEmpty = true;return;}
  if (form.username.length > 10) {error.userOversize = true;return;}
  if (form.useremail == '') {error.emailEmpty = true;return;}
  if (form.useremail.length > 50) {error.emailError = true;return;}
  if (form.userphone == '') {error.phoneEmpty = true;return;}
  if (form.userphone.length != 11) {error.phoneError= true;return;}
  if (form.userpwd == '') {error.pwdEmpty= true;return;}
  if (form.userpwd.length < 7 || form.userpwd.length > 20) {error.pwdOversize= true;return;}

  const pwdRegex = /^[A-Za-z\d@$!%*?&]{7,20}$/;
  const emailRegex = /^[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+$/;
  const phoneRegex = /^1(3[0-9]|5[0-3,5-9]|7[1-3,5-8]|8[0-9])\d{8}$/
  if (!emailRegex.test(form.useremail)) {error.emailError = true;return;}
  if (!pwdRegex.test(form.userpwd)) {error.pwdError = true;return;}
  if (!phoneRegex.test(form.userphone)) {error.phoneError = true;return;}

  let res1 = await axios.get('v2/judgeNameValid',{
    params: {//请求参数（条件查询）
      username:form.username
    }
  });
  if (res1.data['status'] != 200) {error.userExisted=true;return;}

  let hashpwd = sha256(form.userpwd + salt);
  // console.log('sha256',hashpwd);

  let encryptor = new JSEncrypt();
  encryptor.setPublicKey(publicKey);
  let rsapwd = encryptor.encrypt(hashpwd);
  //console.log('rsa',rsapwd)
  let jsonData = JSON.stringify({
    "id":"",
    "username":form.username,
    "password":rsapwd as string,
    "phone":form.userphone,
    "email":form.useremail,
  });
  let res2 = await axios.post('v2/register',jsonData,{
    headers:{'Content-Type':'application/json;'}
  });
  // console.log(res2);
  if (res2.data['status'] == 200) {alert("注册成功");}
  else {alert("注册失败");}
}

const changeType = () =>{
  isLogin.value = !isLogin.value;
  form.username = '';
  form.userpwd = '';
  form.useremail = '';
  form.userphone = '';
}

</script>

<template>
<div class="body">
  <div class="main">
    <div class="big-box" :class="{disactive:!isLogin}">
      <div class="big-contain" key="bigContainLogin" v-if="isLogin">
        <div class="btitle">账户登录</div>
        <div class="bform">
          <input type="email" placeholder="用户名" v-model="form.username">
          <span class="errTips" v-if="error.uesrEmpty">* 请填写用户名 *</span>
          <input type="password" placeholder="密码(7-15位字符)" v-model="form.userpwd">
          <span class="errTips" v-if="error.pwdEmpty">* 请填写密码 *</span>
        </div>
        <button class="bbutton" @click="login">登录</button>
      </div>
      <div class="big-contain" key="bigContainRegister" v-else>
        <div class="btitle">创建账户</div>
        <div class="bform">
          <input type="text" placeholder="用户名(1-10位字符)" v-model="form.username">
          <span class="errTips" v-if="error.uesrEmpty">* 请填写用户名！ *</span>
          <span class="errTips" v-if="error.userOversize">* 用户名超过长度限制！ *</span>
          <span class="errTips" v-if="error.userExisted">* 用户名已经存在！ *</span>
          <input type="email" placeholder="邮箱" v-model="form.useremail">
          <span class="errTips" v-if="error.emailEmpty">* 请填写邮箱！*</span>
          <span class="errTips" v-if="error.emailError">* 请填写正确格式的邮箱！*</span>
          <input type="phone" placeholder="手机号" v-model="form.userphone">
          <span class="errTips" v-if="error.phoneEmpty">* 请填写手机号！*</span>
          <span class="errTips" v-if="error.phoneError">* 请填写正确格式的手机号！*</span>
          <input type="password" placeholder="密码(7-20位字符)" v-model="form.userpwd">
          <span class="errTips" v-if="error.pwdEmpty">* 请填写密码！ *</span>
          <span class="errTips" v-if="error.pwdOversize">* 密码超过长度限制！ *</span>
          <span class="errTips" v-if="error.pwdError">*请填入合法字符的密码！*</span>
        </div>
        <button class="bbutton" @click="register">注册</button>
      </div>
    </div>
    <div class="small-box" :class="{active:isLogin}">
      <div class="small-contain" key="smallContainRegister" v-if="isLogin">
        <div class="stitle">你好，朋友!</div>
        <p class="scontent">开始注册，和我们一起旅行</p>
        <button class="sbutton" @click="changeType">注册</button>
      </div>
      <div class="small-contain" key="smallContainLogin" v-else>
        <div class="stitle">欢迎回来!</div>
        <p class="scontent">与我们保持联系，请登录你的账户</p>
        <button class="sbutton" @click="changeType">登录</button>
      </div>
    </div>
  </div>
</div>
</template>

<style scoped>
.body{
  width: 100vw;
	height: 100vh;
	box-sizing: border-box;
}
.main {
  width: 60%;
  height: 60%;
  position: relative;
  top: 50%;
  left: 50%;
  transform: translate(-50%,-50%);
  border-radius: 20px;
  box-shadow: 0 0 3px #f0f0f0,
        0 0 6px #f0f0f0;
}
.big-box{
  width: 70%;
  height: 100%;
  transform: translateX(0%);
  transition: all 1s;
}
.big-contain{
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}
.btitle{
  font-size: 1.5em;
  font-weight: bold;
  color: rgb(236, 241, 242);
}
.bform{
  width: 100%;
  height: 40%;
  padding: 2em 0;
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  align-items: center;
}
.bform .errTips{
  display: block;
  width: 50%;
  text-align: left;
  color:red;
  font-size: 0.7em;
  display: flex;
  text-align: center;
  justify-content: center;
  border-radius: 20px;
  background-color: #f0f0f0;
}
.bform input{
  width: 50%;
  height: 30px;
  border: none;
  outline: none;
  border-radius: 10px;
  padding-left: 2em;
  background-color: #f0f0f0;
}
.bbutton{
  width: 20%;
  height: 40px;
  border-radius: 24px;
  border: none;
  outline: none;
  background-color: rgb(57,167,176);
  color: #fff;
  font-size: 0.9em;
  cursor: pointer;
}
.small-box{
  width: 30%;
  height: 100%;
  position: absolute;
  box-shadow: 1px 2px 3px #f0f0f0,
        1px 2px 6px #f0f0f0;
  top: 0;
  left: 0;
  transform: translateX(0%);
  transition: all 1s;
  border-top-left-radius: inherit;
  border-bottom-left-radius: inherit;
}
.small-contain{
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}
.stitle{
  font-size: 1.5em;
  font-weight: bold;
  color: #fff;
}
.scontent{
  font-size: 0.8em;
  color: #fff;
  text-align: center;
  padding: 2em 4em;
  line-height: 1.7em;
}
.sbutton{
  width: 60%;
  height: 40px;
  border-radius: 24px;
  border: 1px solid #fff;
  outline: none;
  background-color: transparent;
  color: #fff;
  font-size: 0.9em;
  cursor: pointer;
}

.big-box.disactive{
  position: absolute;
  right: 0;
  transform: translateX(10%);
  transition: all 0.5s;
}
.small-box.active{
  left: 100%;
  border-top-left-radius: 0;
  border-bottom-left-radius: 0;
  border-top-right-radius: inherit;
  border-bottom-right-radius: inherit;
  transform: translateX(-100%);
  transition: all 1s;
}
</style>