package com.example.javaapi.utils;

import com.auth0.jwt.JWTCreator;
import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.interfaces.DecodedJWT;
import com.example.javaapi.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.Calendar;
import java.util.HashMap;

public class JwtUtils {
    @Autowired
    private static UserService userService;

    private static final String SIGN = "kda9y7hbg&&*nwe";
    private static final String ISSUER = "admin";

    public static String createToken(String id){
        HashMap<String, Object> map = new HashMap<>();
        Calendar instance = Calendar.getInstance();
        // 默认1天过期
        instance.add(Calendar.DATE, 1);

        //创建jwt builder
        JWTCreator.Builder builder = JWT.create();

        String token = builder
                .withHeader(map)
                .withIssuer(ISSUER) // 设置发布者
                .withClaim("sub",id)
                .withExpiresAt(instance.getTime())  //指定令牌过期时间
                .sign(Algorithm.HMAC256(SIGN));  // sign
        return token;
    }

    /**
     * 获取id值
     */
    public static String check(String token) {
        try {
            DecodedJWT claims = JWT.require(Algorithm.HMAC256(SIGN))
                    .withIssuer(ISSUER) // 设置发布者
                    .build()
                    .verify(token);
            String id = claims.getSubject();
            return id;
        }catch (Exception e){ // 如果 token 过期会报错 TokenExpiredException
            e.printStackTrace();
            return "error_token";
        }
    }

    /**
     * 验证token  合法性
     */
    public static boolean judge(String token) {
        try {
            DecodedJWT claims = JWT.require(Algorithm.HMAC256(SIGN))
                    .withIssuer(ISSUER) // 设置发布者
                    .build()
                    .verify(token);
            return true;
        }catch (Exception e){ // 如果 token 过期会报错 TokenExpiredException
            e.printStackTrace();
            return false;
        }
    }

}
