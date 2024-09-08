package com.example.javaapi.utils;

import org.mindrot.jbcrypt.BCrypt;

public class BcryptUtils {

    public static String HashData(String decryptedData) throws Exception {
        // 对解密后的数据进行加盐哈希加密
        String hashedData = BCrypt.hashpw(decryptedData,BCrypt.gensalt());
        return hashedData;
    }

    public static boolean CheckPassWord(String rawPwd,String rightPwd){
        boolean valid = BCrypt.checkpw(rawPwd,rightPwd);
        return valid;
    }
}
