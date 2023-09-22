#include <map>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda/xllm-cuda.cuh"

void xllmCudaSetDevice(int gpu_id) {
    cudaSetDevice(gpu_id);
}

struct CudaMemoryBuffer {
    void *data;
    size_t size;
    bool busy;

    CudaMemoryBuffer () {}

    CudaMemoryBuffer (void *data, size_t size, bool busy) :
            data(data), size(size), busy(busy) {}
};

std::map<int, std::vector <CudaMemoryBuffer>> cudaBuffersMap;
std::map<int, std::vector <CudaMemoryBuffer>> bigBuffersMap;

void * xllmCudaMalloc(size_t size) {
    int id = -1;
    cudaGetDevice(&id);
    if (size > 1024 * 1024) {
        auto &bigBuffers = bigBuffersMap[id];
        int selId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].size >= size && !bigBuffers[i].busy
                && bigBuffers[i].size - size < 32 * 1024 * 1024) {
                if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                    selId = i;
                }
            }
        }
        if (selId != -1) {
            bigBuffers[selId].busy = true;
            return bigBuffers[selId].data;
        }

        void * ret;
        cudaMalloc(&ret, size);
        bigBuffers.push_back(CudaMemoryBuffer(ret, size, true));
        return ret;
    }
    auto &cudaBuffers = cudaBuffersMap[id];
    for (int i = 0; i < cudaBuffers.size(); i++) {
        if (cudaBuffers[i].size >= size && !cudaBuffers[i].busy) {
            cudaBuffers[i].busy = true;
            return cudaBuffers[i].data;
        }
    }
    void * ret;
    cudaMalloc(&ret, size);
    cudaBuffers.push_back(CudaMemoryBuffer(ret, size, true));
    return ret;
}

void xllmCudaFree(void *ret) {
    if (ret == nullptr) {
        return;
    }
    for (auto &it : cudaBuffersMap) {
        auto &cudaBuffers = it.second;
        for (int i = 0; i < cudaBuffers.size(); i++) {
            if (cudaBuffers[i].data == ret) {
                cudaBuffers[i].busy = false;
                return;
            }
        }
        auto &bigBuffers = bigBuffersMap[it.first];
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                bigBuffers[i].busy = false;
                return;
            }
        }
    }
    cudaFree(ret);
}

void xllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void xllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void *xllmCudaPrepareInput(const Data &input) {
    void *ret;
    if (input.dataDevice == DataDevice::CUDA) {
        ret = (void*)input.cudaData;
    } else {
        ret = (void*)(input.expansionBytes);
        cudaMemcpy(ret, input.cpuData, input.expansionBytes, cudaMemcpyHostToDevice);
    }
    return ret;
}

void xllmCudaFinishInput(const Data &input, void *data) {
    if (input.dataDevice != DataDevice::CUDA) {
        xllmCudaFree(data);
    }
}

bool xllmCudaRMSNorm(const Data &input, Data &weight, Data &output, float eps) {
    float *cudaInput = (float *) xllmCudaPrepareInput(input);
    float *cudaOutput = (float *) xllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];

    if (channels < 64) {
        FastllmRMSNormKernelInner1<1> <<< outer, 1 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    } else if (channels < 512) {
        FastllmRMSNormKernelInner1<64> <<< outer, 64 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    } else {
        FastllmRMSNormKernelInner1<512> <<< outer, 512 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    }

    xllmCudaFinishInput(input, cudaInput);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}
