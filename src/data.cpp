#include "data.h"

namespace xllm{

    void Data::UpdateUnitSize() {
        if (dataType == DataType::FLOAT32) {
            unitSize = 4;
            unitSizeDiv = 1;
        } else if (dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
            unitSize = 2;
            unitSizeDiv = 1;
        } else if (dataType == DataType::INT8) {
            unitSize = 1;
            unitSizeDiv = 1;
        } else if (dataType == DataType::INT4_NOZERO) {
            unitSize = 1;
            unitSizeDiv = 2;
        }
    }

    Data::Data(DataType type, const std::vector<int> &dims) {
        dataType = type;
        UpdateUnitSize();
        for (int num : dims) {
            counts *= num;
        }
        bytes = (counts * unitSize - 1) / unitSizeDiv + 1;
        Resize(dims);
    }

    void Data::Resize(const std::vector<int> &dims) {
        this->dims = dims;

        strides.resize(dims.size(), 1);
        for (int i = dims.size() - 2; i >= 0; i--) {
            strides[i] = dims[i + 1] * strides[i + 1];
        }
    }

    Data::Data(DataType type, const std::vector<int> &dims, const std::vector<float> &data) : Data::Data(type, dims) {
        Allocate();
        // TODO: 为什么只有FLOAT32 ?
        if (type == DataType::FLOAT32) {
            memcpy(cpuData, data.data(), bytes);
        }
    }

    void Data::Allocate() {
        FreeSpace();
        MallocSpace();
    }

    void Data::MallocSpace() {
        cpuData = new uint8_t[bytes];
    }

    void Data::FreeSpace() {
        delete[] cpuData;
    }

    Data::~Data() {
        delete[] this->cpuData;
    }

    void Data::Reshape(const std::vector<int> &dims) {
        int negative_index = -1;
        uint64_t new_counts = 1;
        for (int i = 0; i < dims.size(); i++) {
            if (dims[i] < 0) {
                if (negative_index == -1) {
                    negative_index = i;
                } else {
                    // dims只能有一个负数索引
                    ErrorInXLLM("Reshape error.\n");
                }
            } else {
                new_counts *= dims[i];
            }
        }
        std::vector <int> outputDims = dims;
        if (negative_index == -1) {
            AssertInXLLM(new_counts == counts, "Reshape error.\n");
        } else {
            AssertInXLLM(new_counts != 0, "Reshape error.\n");
            AssertInXLLM(counts % new_counts == 0, "Reshape error.\n");
            outputDims[negative_index] = counts / new_counts;
        }
        Resize(outputDims);
    }
}