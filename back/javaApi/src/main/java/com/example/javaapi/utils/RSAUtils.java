package com.example.javaapi.utils;

import javax.crypto.Cipher;
import java.nio.charset.StandardCharsets;
import java.security.KeyFactory;
import java.security.PrivateKey;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.Base64;

public class RSAUtils {
    /**
     * 加密算法RSA
     */
    public static final String KEY_ALGORITHM = "RSA";


    /**
     * RSA最大加密明文大小
     */
    private static final int MAX_ENCRYPT_BLOCK = 117;

    /**
     * RSA最大解密密文大小
     */
    private static final int MAX_DECRYPT_BLOCK = 128;
    /**
     * RSA 私钥
     */
    private static final String PRIVATEKEY = "MIICdwIBADANBgkqhkiG9w0BAQEFAASCAmEwggJdAgEAAoGBAK6AUUlQxK0t9qUV" +
            "2Sf1HfPw+/NvK//r/LTuCO4CroyEQJ2dlGt+NoiPs9f42I9LEY9celv+iVWjZUEe45NuBfJgJcoCocHvEBPPEKotb3ZKMtZD2" +
            "kCHEYeUobLfPT7CB6o5x6Bx4buo3Dhvb1I3m1b6DKQN5bJDW29zJ7gWe9xFAgMBAAECgYACRRgJmJveVtU5A4Y+H1Gi7TURzxXm7" +
            "9ZrMhd6pR1JJaVDh5u5JDyoXRotSobiokrZrw3GaZK2wvfbt3+lWM6J+ZXklJtScmKxk9JbZzbtmTEITCGQvUrqo9loUXSqYUk4IfvHL" +
            "72WBvHHhO4qNpgVJlmYwvVIRQ+wLuwRTDNg1wJBAN3IxA9BhTsMhmPOAlZM20h3ytuhyaL9aC7UBb11lcZPsN+HHkdMiqNvyJXbXX6KOL" +
            "JicYfNkgF6kdEE3KlADo8CQQDJbCNhKZFS06qxZgXPh0KI0sVjweMwdNZYufTxcAYWtxj4+9cky51EixPlrz1Q7Sj16Tm24PQpRFbrLqc" +
            "YWRHrAkBpCFR5mSD8hwD6MS0AK+PJnRnQC/3PF82IJ9cUE6S7xy/PnFDlmUUrA5xuA/ZyIAmbyW1U0DEuTBrUb0YFTZXXAkEAnzJfDC5kc" +
            "8hkAu4V7Z6EUcv0wxG9VNEjkOD43dKinVV94Vx7ANQFquUZhtHobovjcekx+n71u6AN6rfmzXGonQJBAKzFEplhl5k0LIa5yB77+gnmcPp" +
            "4B32hHoTDhZX+vWKg6L0N8t8pUUHvPuRnWhfClE4Anr7mxv3x/OKypbsOhTI=";

    /**
    * java 端私钥解密
     */

    public static String decryptRSA(String encryptData) throws Exception{
        if (encryptData == null) return null;
        // 将Base64编码的私钥字符串转换为字节数组
        byte[] privateKeyBytes = Base64.getDecoder().decode(PRIVATEKEY.getBytes(StandardCharsets.UTF_8));

        // 创建PKCS8EncodedKeySpec对象
        PKCS8EncodedKeySpec keySpec = new PKCS8EncodedKeySpec(privateKeyBytes);

        // 获取RSA密钥工厂实例
        KeyFactory keyFactory = KeyFactory.getInstance(KEY_ALGORITHM);

        // 生成私钥对象
        PrivateKey privateKey = keyFactory.generatePrivate(keySpec);

        // 创建RSA解密器
        Cipher cipher = Cipher.getInstance(KEY_ALGORITHM);
        cipher.init(Cipher.DECRYPT_MODE, privateKey);

        // 解密数据
        byte[] decryptedDataBytes = cipher.doFinal(Base64.getDecoder().decode(encryptData.getBytes(StandardCharsets.UTF_8)));

        return new String(decryptedDataBytes, StandardCharsets.UTF_8);
    }
}
