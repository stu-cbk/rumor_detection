package com.example.javaapi.entity;

import lombok.Data;

@Data
public class UserInfo {
    private String id;
    private String username;
    private String password;
    private String phone;
    private String email;
}
