<script setup lang="ts">
import { ref, onMounted, reactive,watch} from 'vue';
import * as echarts from 'echarts';
import {userDataStore} from '@/stores/userData';




const chart = ref()

const userStore = userDataStore();

const tableData = ref(userStore.getRumorList)

const username = ref(userStore.getUsername);
const phone = ref(userStore.getPhone);
const email = ref(userStore.getEmail);


onMounted(()=>{
    var myChart = echarts.init(chart.value)
    const data = [
            { name: "谣言", value: 150 },
            { name: "事实", value: 50 }
        ];

    const option = {
        tooltip: {
      trigger: 'item', // 触发类型为item，表示在圆点上触发显示
      formatter: '{b}: {c}' // 显示的格式，{b}表示类目值，{c}表示数值
    },

    series: [
      {
        type: 'pie',
        radius: ['20%', '60%'],
        // roseType: 'area', //玫瑰图
        data: data,

        label: {
          show: true,
          position: 'outside',
          formatter: '{d}%\n{b}' // 显示百分比和名称
        },
        emphasis: {
          label: {
            show: true,
            fontSize: '20'
            } 
        }
        }
        ]
    };

    myChart.setOption(option);

})

// 监听 pinia 是否发生变化
watch(
    () => [userStore.username,userStore.phone,userStore.email,userStore.isRumor,
    userStore.notRumor,userStore.rumorList],
    ([new_username,new_phone,new_email,new_isRumor,new_notRumor,new_texts],
    [old_username,old_phone,old_email,old_isRumor,old_notRumor,old_texts]) =>{
        username.value = new_username as string;
        phone.value = new_phone as string;
        email.value = new_email as string;

        tableData.value = new_texts as Array<any>;
        console.log(tableData)
        const newDataRate = [
            { name: "谣言", value: new_isRumor as number },
            { name: "事实", value: new_notRumor as number }
        ];

        var myChart = echarts.init(chart.value)

        myChart.setOption({
            series: [{
                data: newDataRate
            }]
        });
    }
);

</script>

<template>
    <div class="main">
        <el-container class="main">
            <el-header class="main-header">
                <el-card class="main-header">
                <el-row :gutter="4">
                    <el-col :span="3">
                        <el-avatar
                        src="https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png"
                        :size="100"
                        />
                    </el-col>
                    <el-col :span="18">
                    <el-descriptions title="用户信息">
                        <el-descriptions-item label="Username":span="2" >{{ username }}</el-descriptions-item>
                        <el-descriptions-item label="Telephone" :span="2" >{{ phone  }}</el-descriptions-item>
                        <el-descriptions-item label="email" :span="2" >{{ email }}</el-descriptions-item>
                    </el-descriptions>
                    </el-col>
                </el-row>
                </el-card>
            </el-header>
            <el-main class="main-main">
                <el-container class="main-main">
                    <el-aside class="main-left">
                        <el-card class="left" style="width: 600px;height:400px;">
                            <div ref="chart" style="width: 400px;height:300px;"></div>
                        </el-card>
                    </el-aside>
                    <el-main  class="main-right">
                        <el-scrollbar class="right">
                            <el-table :data="tableData">
                            <el-table-column prop="isRumor" label="类别" />
                            <el-table-column prop="rumorText" label="详情" width="700px"/>
                            </el-table>
                        </el-scrollbar>
                    </el-main>
                </el-container>
            </el-main>
        </el-container>
    </div>
</template>
<style scoped>
.main{
    height: 100%;
    width: 100%;
    display: flex;
    justify-content:center;
    align-items: center;
}
.main-header{
    height: 150px;
    width: 100%;
}
.main-main{
    width: 100%;
    height: 440px;
}
.main-left{
    height: 100%;
    width: 50%;
    display: flex;
    justify-content:center;
    align-items: center;
}
.main-right{
    height: 100%;
    width: 50%;
    display: flex;
    justify-content:center;
    align-items: center;
}
.left{
    display: flex;
    justify-content:center;
    align-items: center;
}
.right{
    border-radius: 4px;
    height: 95%;
    width: 100%;
}
</style>