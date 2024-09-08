package com.example.javaapi;


import com.auth0.jwt.JWT;
import com.auth0.jwt.JWTCreator;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.interfaces.DecodedJWT;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.example.javaapi.entity.RumorInfo;
import com.example.javaapi.service.RumorService;
import com.example.javaapi.utils.JwtUtils;
import com.example.javaapi.utils.Result;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.File;
import java.io.IOException;
import java.util.*;

@SpringBootTest

class JavaApiApplicationTests {
    @Autowired
    RumorService rumorService;

    @Test
    public void testRumorInsert() {
        // 创建ObjectMapper对象
        ObjectMapper objectMapper = new ObjectMapper();
       String filePath = "D:\\bishe\\examdata\\data2.json";
        try {
            List<RumorInfo> dataList = objectMapper.readValue(new File(filePath), new TypeReference<List<RumorInfo>>() {});
            for (RumorInfo rumor:dataList){
                String id = IdWorker.getIdStr();
                rumor.setId(id);
                rumorService.addOneData(rumor);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testRumorSelect(){
        try{
            List<RumorInfo> rumorList = rumorService.selectAllData();
            int cnt = 0,all = rumorList.size();
            for (int i = 0;i < all;i ++)
                if (rumorList.get(i).getIsRumor().equals("1"))
                    cnt = cnt + 1;
            List<Integer> res = new ArrayList<>();
            res.add(all - cnt);
            res.add(cnt);
            System.out.println(res);

        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
