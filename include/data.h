#pragma once
#include <vector>
#include <map>
#include <string>
#include <cstring>
#include <iostream>
#include <cmath>

#include "file.h"
#include "utils.h"
#include "xllm.h"

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
        uint64_t bytes = 0; // 字节数
        // 跨度, 用于快速计算指定维度的元素数量
        // 维度与dims相同，strides[i]=dims[i+1]*dims[i+2]*...*1
        // 例如：dims(2,3,4) -> strides(12,4,1), dims(2,3) -> strides(3,1)
        std::vector <uint64_t> strides;

        uint64_t expansionSize = 0; // 扩容后的尺寸
        uint64_t expansionBytes = 0; // 扩容后的字节数
        std::vector <int> expansionDims;  // 预扩容的数据形状

        uint8_t *cpuData = nullptr; // 数据指针
        int assignBytes = 0;     // 已分配的字节数

        Data() {};
        Data (DataType type);
        Data (DataType type, const std::vector <int> &dims);
        Data (DataType type, const std::vector <int> &dims, const std::vector <float> &data);

        ~Data(); 

        Data (const Data &ori); // 深拷贝
        void CopyFrom(const Data &ori); // 复制
        
        void Allocate(); // 分配内存
        void MallocSpace(int bytes); // 在设备上分配
        void FreeSpace(); // 回收设备上的内存

        void UpdateUnitSize(); // 更新unitSize

        void Resize(const std::vector <int> &dims); // 更改尺寸

        void Reshape(const std::vector <int> &dims); // 更改尺寸, 但不移动数据

        uint64_t Count(int i) const;

        void Expansion(const std::vector <int> &dims); // 预扩容到相应尺寸

        void PrintShape() const; // 输出形状
        void Print() const; // 输出

        void CalcWeightSum(); // 计算WeightSum
    };
}
