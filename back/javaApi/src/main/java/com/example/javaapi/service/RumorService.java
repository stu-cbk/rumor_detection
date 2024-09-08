package com.example.javaapi.service;

import com.example.javaapi.entity.RumorInfo;

import java.util.List;

public interface RumorService {
    List<RumorInfo> selectAllData();
    Integer addOneData(RumorInfo rumor);
}
