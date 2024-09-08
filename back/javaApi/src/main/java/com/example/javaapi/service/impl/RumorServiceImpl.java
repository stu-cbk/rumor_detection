package com.example.javaapi.service.impl;

import com.example.javaapi.entity.RumorInfo;
import com.example.javaapi.mapper.RumorMapper;
import com.example.javaapi.service.RumorService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class RumorServiceImpl implements RumorService {
    @Autowired
    private RumorMapper rumorMapper;

    public List<RumorInfo> selectAllData(){return rumorMapper.selectAllData();}
    public Integer addOneData(RumorInfo rumor){return rumorMapper.addOneData(rumor);}
}
