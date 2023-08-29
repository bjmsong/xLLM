#pragma once
#include <vector>
#include <map>
#include <string>
#include <cstring>
#include <iostream>
#include <cmath>
#include "file.h"
#include "utils.h"

namespace xllm{

enum DataType {
    FLOAT32 = 0, BFLOAT16 = 1, INT8 = 3, INT4_NOZERO = 8, FLOAT16 = 7
};

enum WeightType {
    NONE = 0, LINEAR = 1, EMBEDDING = 2
};

class Data {
    public:
        WeightType weightType = WeightType::NONE; // 权重类型，NONE代表非权重（或未知权重）

        DataType dataType = DataType::FLOAT32; // 数据类型
        int unitSize = 4;         // dataType占几个字节
        int unitSizeDiv = 1;  // unitSIze / unitSizeDiv: 单个元素占几个字节

        std::vector <int> dims; // 数据形状
        int counts = 1; // 元素数量
        uint64_t bytes = 0; // 元素字节数

        // 跨度, 维度与dims相同，strides[i]=dims[i+1]*dims[i+2]*...*1
        // dims(2,3,4) -> strides(12,4,1), dims(2,3) -> strides(3,1)
        std::vector <uint64_t> strides;

        uint8_t *cpuData = nullptr; // 数据指针
        int assignBytes = 0;     // 已经分配的空间

        Data() {};
        Data (DataType type);
        Data (DataType type, const std::vector <int> &dims); // 构造函数
        // data中是原始数据，如果type不是float那么需要量化
        Data (DataType type, const std::vector <int> &dims, const std::vector <float> &data);

        ~Data(); 

        // Data (const Data &ori); // 深拷贝

        void Allocate(); // 分配内存

        void Expansion(const std::vector <int> &dims); // 预扩容到相应尺寸

        void MallocSpace(); // 在设备上分配

        void FreeSpace(); // 回收设备上的内存

        void UpdateUnitSize(); // 更新unitSize

        void Resize(const std::vector <int> &dims); // 更改尺寸

        void Reshape(const std::vector <int> &dims); // 更改尺寸,但不修改数据

        void PrintShape() const; // 输出形状

        void Print() const; // 输出

        void CalcWeightSum(); // 计算WeightSum
    };
}
