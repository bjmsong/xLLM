#include "cuda/cudadevice.h"
#include "cuda/xllm-cuda.cuh"

namespace xllm {
    CudaDevice::CudaDevice() {
        this->deviceType = "cuda";
        this->ops["LayerNorm"] = (BaseOperator*)(new CudaLayerNormOp());
        this->ops["RMSNorm"] = (BaseOperator*)(new CudaRMSNormOp());
        this->ops["Linear"] = (BaseOperator*)(new CudaLinearOp());
        this->ops["CatDirect"] = (BaseOperator*)(new CudaCatDirectOp());
        this->ops["MatMul"] = (BaseOperator*)(new CudaMatMulOp());
        this->ops["MatMulTransB"] = (BaseOperator*)(new CudaMatMulTransBOp());
        this->ops["SoftMax"] = (BaseOperator*)(new CudaSoftMaxOp());
        this->ops["Silu"] = (BaseOperator*)(new CudaSiluOp());
        this->ops["AddTo"] = (BaseOperator*)(new CudaAddToOp());
        this->ops["MulTo"] = (BaseOperator*)(new CudaMulToOp());
        this->ops["AttentionMask"] = (BaseOperator*)(new CudaAttentionMaskOp());
        this->ops["PermuteSelf"] = (BaseOperator*)(new CudaPermuteSelfOp());
        this->ops["LlamaRotatePosition2D"] = (BaseOperator*)(new CudaLlamaRotatePosition2DOp());
    }

    bool CudaDevice::Malloc(void **ret, size_t size) {
        *ret = xllmCudaMalloc(size);
        return true;
    }

    bool CudaDevice::Free(void *ret) {
        xllmCudaFree(ret);
        return true;
    }

    bool CudaDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        xllmCudaCopyFromHostToDevice(dst, src, size);
        return true;
    }

    bool CudaDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        xllmCudaCopyFromDeviceToHost(dst, src, size);
        return true;
    }

    void CudaRMSNormOp::Run(const std::string &opType, const DataDict &datas,
                       const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-5;
        xlmCudaRMSNorm(input, weight, output, eps);
    }

    bool CudaLayerNormOp::CanRun(const std::string &opType, const DataDict &datas,
                                 const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int inner = input.strides[axis];
        return inner == 1;
    }

    void CudaLayerNormOp::Run(const std::string &opType, const DataDict &datas,
                             const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &gamma = *(datas.find("gamma")->second);
        Data &beta = *(datas.find("beta")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        output.Allocate();
        FastllmCudaLayerNorm(input, gamma, beta, output, axis);
    }

    void CudaLinearOp::Run(const std::string &opType, const DataDict &datas,
                           const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate();
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();

        if (weight.dataType == DataType::FLOAT32) {
            FastllmCudaMatMulFloat32(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::FLOAT16) {
            FastllmCudaMatMulFloat16(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::INT8) {
            FastllmCudaMatMulFloatInt8(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::INT4) {
            FastllmCudaMatMulFloatInt4(input, weight, bias, output, n, m, k);
        } else if (weight.dataType == DataType::INT4_NOZERO) {
            FastllmCudaMatMulFloatInt4NoZero(input, weight, bias, output, n, m, k);
        } else {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        }
    }

    void CudaCatDirectOp::Run(const std::string &opType, const DataDict &datas,
                             const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        AssertInFastLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "Cat's input's type should be float32.\n");
        AssertInFastLLM(input0.dataDevice == input1.dataDevice, "CatDirect error: inputs should use same device.\n");

        if (input0.dims.size() == 0) {
            input0.Resize(input1.dims);
            AssertInFastLLM(input0.expansionDims.size() == input1.dims.size() &&
                            input1.dims[axis] <= input0.expansionDims[axis],
                            "CatDirect Error: input0's expansion size is not enough.\n");
            int outer = input1.Count(0) / input1.Count(axis);
            int input0Stride = input0.Count(axis);
            int input1Stride = input1.Count(axis);
            int inner = input0.strides[axis];
            int unitSize = input0.unitSize;
            FastllmCudaMemcpy2DDeviceToDevice((uint8_t *) input0.cudaData, input0Stride * unitSize,
                                              (uint8_t *) input1.cudaData, input1Stride * unitSize,
                                              input1.dims[axis] * inner * unitSize, outer);
            return;
        }

        AssertInFastLLM(input0.dims.size() == input1.dims.size(), "Cat Error: input's shape's size should be same.\n");
        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        for (int i = 0; i < dimsLen; i++) {
            if (i != axis) {
                AssertInFastLLM(input0.dims[i] == input1.dims[i], "Cat Error: input's shape doesn't match.");
            }
        }

        std::vector<int> dims = input0.dims;
        std::vector<int> oldDims = dims;
        dims[axis] += input1.dims[axis];
        input0.Resize(dims);
        int outer = input0.Count(0) / input0.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);

        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        FastllmCudaMemcpy2DDeviceToDevice((uint8_t *) input0.cudaData + oldDims[axis] * inner * unitSize,
                                          input0Stride * unitSize,
                                          (uint8_t *) input1.cudaData, input1Stride * unitSize,
                                          input1.dims[axis] * inner * unitSize, outer);
    }

    void CudaMatMulOp::Run(const std::string &opType, const DataDict &datas,
                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : -1;
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 1];
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2);
        FastllmCudaBatchMatMul(input0, input1, output,
                     input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                     batch0, n, m, k, alpha);
    }

    void CudaMatMulTransBOp::Run(const std::string &opType, const DataDict &datas,
                                const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : -1;
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 2];
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2);
        FastllmCudaBatchMatMulTransB(input0, input1, output,
                     input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                     batch0, n, m, k, alpha);
    }

    bool CudaSoftMaxOp::CanRun(const std::string &opType, const DataDict &datas,
                               const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int inner = input.Count(axis + 1);
        if (inner != 1) {
            return false;
        }
        return true;
    }

    void CudaSoftMaxOp::Run(const std::string &opType, const DataDict &datas,
                            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Softmax error: Data's type should be float32.\n");
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        FastllmCudaSoftmax(input, output, axis);
    }

    void CudaSiluOp::Run(const std::string &opType, const DataDict &datas,
                            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Silu error: Data's type should be float32.\n");
        FastllmCudaSilu(input, output);
    }

void CudaAddToOp::Run(const std::string &opType, const DataDict &datas,
                         const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;

        AssertInFastLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "AddTo error: Data's type should be float32.\n");
        AssertInFastLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");
        FastllmCudaAddTo(input0, input1, alpha);
    }

    void CudaMulToOp::Run(const std::string &opType, const DataDict &datas,
                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;

        AssertInFastLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "MulTo error: Data's type should be float32.\n");
        AssertInFastLLM(input0.dims == input1.dims, "MulTo error: input's shape should be same.\n");
        FastllmCudaMulTo(input0, input1, alpha);
    }

    void CudaAttentionMaskOp::Run(const std::string &opType, const DataDict &datas,
                                  const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
        FastllmCudaAttentionMask(input, mask, maskValue);
    }

void CudaPermuteSelfOp::Run(const std::string &opType, const DataDict &datas,
                               const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &axisData = *(datas.find("axis")->second);
        std::vector <int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }

        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Permute error: datatype should be float32.");
        AssertInFastLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");

        bool same = false;
        same |= ((axis == std::vector <int>{1, 2, 0} || axis == std::vector <int>{1, 0, 2}) && (input.dims[0] == 1 || input.dims[1] == 1));
        same |= ((axis == std::vector <int>{2, 0, 1, 3}) && input.dims[2] == 1);
        same |= ((axis == std::vector <int>{0, 2, 1, 3}) && (input.dims[1] == 1 || input.dims[2] == 1));
        if (same) {
            std::vector<int> new_dims;
            for (int i = 0; i < axis.size(); i++) {
                new_dims.push_back(input.dims[axis[i]]);
            }
            input.Resize(new_dims);
            return;
        }

        FastllmCudaPermute(input, axis);
    }

    void CudaLlamaRotatePosition2DOp::Run(const std::string &opType, const DataDict &datas,
                                     const FloatDict &floatParams, const IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;

        FastllmCudaLlamaRotatePosition2D(data, positionIds, sinData, cosData, rotaryDim);
    }
}
