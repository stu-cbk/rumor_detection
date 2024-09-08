package com.example.javaapi.controller;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.example.javaapi.entity.RumorInfo;
import com.example.javaapi.entity.UserInfo;
import com.example.javaapi.service.RumorService;
import com.example.javaapi.service.UserService;
import com.example.javaapi.utils.BcryptUtils;
import com.example.javaapi.utils.JwtUtils;
import com.example.javaapi.utils.RSAUtils;
import com.example.javaapi.utils.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

@RestController
@CrossOrigin(origins = "http://127.0.0.1:5173")

public class UserController {
    @Autowired
    private UserService userService;
    @Autowired
    private RumorService rumorService;
    @GetMapping("/judgeNameValid")
    public Result selectByName(@RequestParam("username")  String username)
    {
        if (username.equals(""))
            return Result.fail("用户名不能为空");
        UserInfo user = userService.selectByName(username);
        if (user == null){
            return Result.suc("用户不存在");
        }else{
            return Result.fail("用户已存在");
        }
    }

    @PostMapping("/register")
    public Result register(@RequestBody UserInfo user)
    {
        // 此时前端传过来的password是经过加密了的 这还需要解密
        try {
            if (user.getPassword().equals("") || user.getUsername().equals(""))
                return Result.fail("登录失败");
            // 首先解密出前端传过来的密码
            String userPwd = RSAUtils.decryptRSA(user.getPassword());
            // 然后对这个密码进行二次加密
            String hashData = BcryptUtils.HashData(userPwd);
            user.setPassword(hashData);
            user.setId(IdWorker.getIdStr());
            Integer isSuccess = userService.addOneData(user);
            String id = user.getId();
            // System.out.println("isSuccess:" + isSuccess);
            System.out.println("主键 ID: " + id);
            if (isSuccess.equals(1))
                return Result.suc("注册成功");
            else
                return Result.fail("注册失败");
        } catch (Exception e) {
            e.printStackTrace();
            return Result.fail("操作失败");
        }
    }

    @PostMapping("/login")
    public Result login(@RequestBody UserInfo user)
    {
        // 此时前端传过来的password是经过加密了的 这还需要解密
        try {
            if (user.getPassword().equals("")|| user.getUsername().equals(""))
                return Result.fail("登录失败");
            // 首先解密出前端传过来的密码
            String userPwd = RSAUtils.decryptRSA(user.getPassword());
            // 然后根据用户名找到密码
            UserInfo rightUser = userService.selectByName(user.getUsername());
            String rightPwd = rightUser.getPassword();
            if (BcryptUtils.CheckPassWord(userPwd,rightPwd)) {
                String token = JwtUtils.createToken(rightUser.getId());
                return Result.suc(token);
            }else
                return Result.fail("登录失败");
        } catch (Exception e) {
            e.printStackTrace();
            return Result.fail("操作失败");
        }
    }

    @GetMapping("/getUserData")
    public Result getUserData(@RequestParam("token")  String token){
        String id = JwtUtils.check(token);
        if (id.equals("error_token")) {return Result.fail("查询失败");}
        UserInfo userData = userService.selectByID(id);
        if (userData == null) {return Result.fail("查询失败");}
        userData.setPassword("");
        userData.setId("");
        return Result.suc(userData);
    }

    @GetMapping("/getRate")
    public Result getRate(){
        try{
            List<RumorInfo> rumorList = rumorService.selectAllData();
            if (rumorList == null) return Result.fail("操作失败");
            int cnt = 0,all = rumorList.size();
            for (int i = 0;i < all;i ++)
                if (rumorList.get(i).getIsRumor().equals("1"))
                    cnt = cnt + 1;
            List<Integer> res = new ArrayList<>();
            res.add(all - cnt);
            res.add(cnt);

            return Result.suc(res);
        }catch (Exception e){
            e.printStackTrace();
            return Result.fail("操作失败");
        }
    }

    @GetMapping("/getRumorList")
    public Result getRumor(){
        try{
            List<RumorInfo> rumorList = rumorService.selectAllData();
            if (rumorList == null) return Result.fail("操作失败");
            List<RumorInfo> res = new ArrayList<>();
            for (int i = 0;i < 50;i ++)
            {
                RumorInfo r1 = rumorList.get(i);
                RumorInfo r2 = new RumorInfo();
                if (r1.getIsRumor().equals("1"))
                    r2.setIsRumor("谣言");
                else
                    r2.setIsRumor("事实");
                String text = r1.getRumorText();
                if (text.length() >= 35) text = text.substring(0,32) + "...";
                r2.setRumorText(text);
                res.add(r2);
            }
            return Result.suc(res);
        }catch (Exception e){
            e.printStackTrace();
            return Result.fail("操作失败");
        }
    }
}
