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
        } else if (dataType == DataType::INT4) {
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
        if (type == DataType::FLOAT32) {
            std::memcpy(cpuData, data.data(), bytes);
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

}