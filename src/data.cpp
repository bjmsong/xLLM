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

    Data::Data(DataType type) {
        dataType = type;
        UpdateUnitSize();
    }

    Data::Data(DataType type, const std::vector<int> &dims) : Data::Data(type) {
        counts = 1;
        for (int num : dims) {
            counts *= num;
        }
        bytes = (counts * unitSize - 1) / unitSizeDiv + 1;
        Resize(dims);
    }

    void Data::Resize(const std::vector<int> &dims) {
        this->dims = dims;
        
        strides.clear();
        strides.resize(dims.size(), 1);
        for (int i = dims.size() - 2; i >= 0; i--) {
            strides[i] = dims[i + 1] * strides[i + 1];
        }
    }

    Data::Data(DataType type, const std::vector<int> &dims, const std::vector<float> &data) : Data::Data(type, dims) {
        Allocate();
        // 如果不是float32那么需要量化
        if (type == DataType::FLOAT32) {
            memcpy(cpuData, data.data(), bytes);
        }
    }

    void Data::Allocate() {
        if(assignBytes<bytes){
            FreeSpace();
            MallocSpace(bytes);
        }
    }

    void Data::FreeSpace() {
        delete[] cpuData;
    }

    void Data::MallocSpace(uint64_t bytes) {
        cpuData = new uint8_t[bytes];
        
        assignBytes = bytes;
    }

    Data::~Data() {
        delete[] this->cpuData;
    }

    Data::Data(const Data &ori) {
        CopyFrom(ori);
    }

    void Data::CopyFrom(const Data &ori) {
        if (ori.dims != this->dims || this->cpuData == nullptr) {
            if (ori.dims.size() == 0) {
                delete[] this->cpuData;
                this->dataType = ori.dataType;
                this->UpdateUnitSize();
                this->dims.resize(0);
                this->cpuData = nullptr;
                return;
            }
            this->dataType = ori.dataType;
            this->Resize(ori.dims);
            counts = 1;
            for (int num : dims) {
                counts *= num;
            }
            this->bytes = (counts * unitSize - 1) / unitSizeDiv + 1;
            this->Allocate();
        }
        std::memcpy(this->cpuData, ori.cpuData, bytes);
    }

    void Data::Reshape(const std::vector<int> &dims) {
        int negative_index = -1;
        uint64_t new_counts = 1;
        for (int i = 0; i < dims.size(); i++) {
            if (dims[i] < 0) {
                if (negative_index == -1) {
                    negative_index = i;
                } else {
                    // dims只能包含一个负数索引
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

    uint64_t Data::Count(int i)  const{
        if (i >= this->dims.size()) {
            return 1;
        }
        if (i - 1 >= 0 && i - 1 < this->strides.size()) {
            return this->strides[i - 1];
        }
        return this->dims[i] * this->strides[i];
    }

    void Data::Expansion(const std::vector<int> &dims) {
        
        expandDims = dims;
        expandCounts = 1;
        for (int dim : dims) {
            expandCounts *= dim;
        }
        expandBytes = (expandCounts * unitSize - 1) / unitSizeDiv + 1;

        if (this->dims.size() == 0 || assignBytes == 0) {
            this->MallocSpace(expandBytes);
            return;
        }

        AssertInXLLM(dims.size() == this->dims.size(), "Expansion error: real dims's size should equal to expansion dims's size.\n");
        for (int i = 0; i < dims.size(); i++) {
            AssertInXLLM(dims[i] == -1 || dims[i] >= this->dims[i], "Expansion error: real size should <= expansion size.\n");
        }

        MallocSpace(expandBytes);

        // 要扩张哪一个维度
        int axis = -1;
        for (int i = 0; i < dims.size(); i++) {
            if (this->dims[i] < dims[i]) {
                axis = i;
                break;
            }
        }

        int inputStride = this->Count(axis);
        uint8_t *old = this->cpuData;
        // 把原来的数据拷贝到新的空间
        int outer = this->counts / inputStride;
        for (int o = 0; o < outer; o++) {
            memcpy(this->cpuData + o * inputStride/this->dims[axis]*dims[axis] * unitSize,
                    old + o * inputStride * unitSize,
                    inputStride * unitSize);
        }
        delete[] old;
    }    
}
