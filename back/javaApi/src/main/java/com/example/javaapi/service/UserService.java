package com.example.javaapi.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.example.javaapi.entity.UserInfo;
import com.example.javaapi.service.impl.UserServiceImpl;

import java.util.List;

public interface UserService{
    UserInfo selectByName(String username);
    UserInfo selectByID(String id);
    Integer addOneData(UserInfo user);
}
