package com.example.javaapi.mapper;

import com.example.javaapi.entity.UserInfo;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface UserMapper {
    UserInfo selectByName(String username);
    UserInfo selectByID(String id);
    List<UserInfo> selectAllData();
    Integer addOneData(UserInfo user);
    Integer deleteByName(String username);
    Integer updateOneData(UserInfo user);
}
