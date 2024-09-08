package com.example.javaapi.mapper;

import com.example.javaapi.entity.RumorInfo;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface RumorMapper {
    List<RumorInfo> selectAllData();
    Integer addOneData(RumorInfo rumor);
}
