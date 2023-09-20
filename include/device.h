#pragma once

#include "xllm.h"
#include "data.h"

namespace xllm {
    typedef std::map <std::string, Data*> DataDict;
    typedef std::map <std::string, float> FloatDict;
    typedef std::map <std::string, int> IntDict;

    class BaseOperator {
    public:
        // 是否可以运行某一个算子
        virtual bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

        // 对某一个算子进行推理
        virtual void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) = 0;
    };

    class BaseDevice {
    public:
        virtual bool Malloc (void **ret, size_t size) = 0; // 分配尺寸为size的空间
        virtual bool Malloc (void **ret, Data &data); // 分配形状为dims的空间
        virtual bool Free(void *ret) = 0; // 释放ret

        virtual bool CopyDataToCPU(void *dst, void *src, size_t size) = 0; // device上的src拷贝到cpu上的dst
        virtual bool CopyDataToCPU(Data &data); // data数据从该device移动到CPU

        virtual bool CopyDataFromCPU(void *dst, void *src, size_t size) = 0; // cpu上的src拷贝到device上的dst
        virtual bool CopyDataFromCPU(Data &data); // data数据从CPU移动到该device

        // 是否可以运行某一个算子
        virtual bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

        // 对某一个算子进行推理
        virtual void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

        std::string deviceType;
        std::string deviceName;
        std::vector <int> deviceIds;

        std::map <std::string, BaseOperator*> ops;
    };
}