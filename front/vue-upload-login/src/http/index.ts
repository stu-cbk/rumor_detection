import axios from 'axios'

// 配置路径
axios.defaults.baseURL = '/api/';

// 请求拦截
axios.interceptors.request.use(config=>config);
// 响应拦截
axios.interceptors.response.use(
    res=>res,
    err=>Promise.reject(err)
);

export default axios
