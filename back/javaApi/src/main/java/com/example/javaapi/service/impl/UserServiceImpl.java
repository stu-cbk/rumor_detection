package com.example.javaapi.service.impl;


import com.example.javaapi.entity.UserInfo;
import com.example.javaapi.mapper.UserMapper;
import com.example.javaapi.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class UserServiceImpl implements UserService{
    @Autowired
    private UserMapper userMapper;

    @Override
    public UserInfo selectByName(String username){return userMapper.selectByName(username);}

    @Override
    public Integer addOneData(UserInfo user) {return userMapper.addOneData(user);}

    @Override
    public UserInfo selectByID(String id) {return userMapper.selectByID(id);}
}
