#include <map>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda/xllm-cuda.cuh"

// #define CHECK(call)
// {
//     const cudaError_t error = call;
//     if (error != cudaSuccess)
//     { 
//         printf("Error: %s:%d, ", __FILE__, __LINE__);
//         printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
//         exit(1);
//     }
// }

static std::map<int, cublasHandle_t> s_xllmCublasHandleMap;
cublasHandle_t getxllmCublasHandle() {
    int id = -1;
    cudaGetDevice(&id);
    auto it = s_xllmCublasHandleMap.find(id);
    if (it != s_xllmCublasHandleMap.end()) {
        return it->second;
    }
    cublasHandle_t handler = nullptr;
    auto stat = cublasCreate(&handler);
    // cudaDeviceSynchronize();

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed:%d\n", stat);
        exit(0);
    } else {
        s_xllmCublasHandleMap[id] = handler;
    }

    return handler;
}


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
        // cudaError_t cudaStatus = cudaDeviceSynchronize();
        // if (cudaStatus != cudaSuccess) {
        //     fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // }
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
    // cudaError_t cudaStatus = cudaDeviceSynchronize();
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
    // }
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

void xllmCudaClearBigBuffer() {
    int id = -1;
    cudaGetDevice(&id);
    for (auto &it : bigBuffersMap) {
        auto &bigBuffers = it.second;
        std::vector <CudaMemoryBuffer> temp;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy) {
                cudaSetDevice(it.first);
                cudaFree(bigBuffers[i].data);
            } else {
                temp.push_back(bigBuffers[i]);
            }
        }
        bigBuffers.clear();
        bigBuffers = temp;
    }
    cudaSetDevice(id);
}


void xllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void xllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void *xllmCudaPrepareInput(const xllm::Data &input) {
    void *ret;
    if (input.dataDevice == xllm::DataDevice::CUDA) {
        ret = (void*)input.cudaData;
    } else {
        ret = (void*)(input.assignBytes);
        cudaMemcpy(ret, input.cpuData, input.assignBytes, cudaMemcpyHostToDevice);
    }
    return ret;
}

void xllmCudaFinishInput(const xllm::Data &input, void *data) {
    if (input.dataDevice != xllm::DataDevice::CUDA) {
        xllmCudaFree(data);
    }
}

void *xllmCudaPrepareOutput(xllm::Data &output) {
    void *ret;
    if (output.dataDevice == xllm::DataDevice::CUDA) {
        ret = (float*)output.cudaData;
    } else {
        ret = (float*)xllmCudaMalloc(output.assignBytes);
    }
    return ret;
}

void xllmCudaFinishOutput(xllm::Data &output, void *data) {
    if (output.dataDevice != xllm::DataDevice::CUDA) {
        cudaMemcpy(output.cpuData, data, output.assignBytes, cudaMemcpyDeviceToHost);
        xllmCudaFree(data);
    }
    // cudaDeviceSynchronize();
}

void xllmCudaMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height) {
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice);
    // cudaDeviceSynchronize();
}

template <int THREAD_PER_BLOCK>
__global__ void xllmRMSNormKernelInner1(float *input, float *weight, float *output, int outer, int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float scale;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum2 = 0.0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = input[i];
        sum2 += x * x;
    }
    sdata2[tid] = sum2;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    // 3. 计算参数
    if (tid == 0) {
        scale = 1.0 / sqrt(sdata2[0] / channels + eps);
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = (input[i] * scale * weight[i]);
    }
}

bool xllmCudaRMSNorm(const xllm::Data &input, xllm::Data &weight, xllm::Data &output, float eps) {
    float *cudaInput = (float *) xllmCudaPrepareInput(input);
    float *cudaOutput = (float *) xllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];

    if (channels < 64) {
        xllmRMSNormKernelInner1<1> <<< outer, 1 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    } else if (channels < 512) {
        xllmRMSNormKernelInner1<64> <<< outer, 64 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    } else {
        xllmRMSNormKernelInner1<512> <<< outer, 512 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    }

    xllmCudaFinishInput(input, cudaInput);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}

__global__ void xllmCudaFloat2HalfKernel(float* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2half(a[idx]);
    }
}

__global__ void xllmCudaHalf2FlotaKernel(half* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __half2float(a[idx]);
    }
}

__global__ void xllmCudaBiasKernel(float *a, float *bias, int k) {
    float *now = a + blockIdx.x * k;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
        now[i] += bias[i];
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp32Fp16Kernel2(float *A, half *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * (float)B[p * m + i];
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] + bias[p];
        }
        __syncthreads();
    }
}

bool xllmCudaMatMulFloat16(const xllm::Data &input, xllm::Data &weight, const xllm::Data &bias, xllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaBiasData;
        cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }
    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)xllmCudaPrepareInput(input);
    float *cudaOutput = (float*)xllmCudaPrepareOutput(output);

    if (n > 1) {
        half *cudaFp16Input, *cudaFp16Output;
        cudaFp16Input = (half *) xllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) xllmCudaMalloc(n * k * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        auto fastllmCublasHandle = getxllmCublasHandle();
        // cudaDeviceSynchronize();
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        xllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, (half *) weight.cudaData, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
        xllmCudaHalf2FlotaKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        xllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
        // cudaDeviceSynchronize();

        xllmCudaFree(cudaFp16Input);
        xllmCudaFree(cudaFp16Output);
    } else {
        FastllmGemvFp32Fp16Kernel2<256, 1> <<< k, 256 >>>(cudaInput, (half *) weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }

    xllmCudaFinishInput(input, cudaInput);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}

__global__ void xllmLlamaRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + m / 2];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + m / 2] = va * curSin + vb * curCos;
}


bool xllmCudaLlamaRotatePosition2D(xllm::Data &data, const xllm::Data &positionIds,
                                      const xllm::Data &sinData, const xllm::Data &cosData, int rotaryDim) {
    float *cudaData = (float *) xllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) xllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) xllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) xllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    xllmLlamaRotatePosition2DKernel <<< outer * n, std::min(rotaryDim, m / 2) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    xllmCudaFinishInput(positionIds, cudaPositionIds);
    xllmCudaFinishInput(sinData, cudaSin);
    xllmCudaFinishInput(cosData, cudaCos);
    xllmCudaFinishOutput(data, cudaData);
    return true;
}

template <int THREAD_PER_BLOCK>
__global__ void xllmTransposeByRowKernel(uint8_t *dst, uint8_t *ori, int n, int m, int k) {
    int row = blockIdx.x / m, col = blockIdx.x % m;
    uint8_t *curInput = ori + (row * m + col) * k;
    uint8_t *curOutput = dst + (col * n + row) * k;
    for (int i = threadIdx.x; i < k; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

__global__ void xllmPermuteKernel(float *dst, float *ori, int *temp, int axisLen, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        int old = 0;
        int idx = i;
        for (int j = 0; j < axisLen; ++j) {
            int order = temp[j];
            old += (idx / temp[j + 2 * axisLen]) * temp[order + 1 * axisLen];
            idx %= temp[j + 2 * axisLen];
        }
        dst[i] = ori[old];
    }
}

bool xllmCudaPermute(xllm::Data &input, const std::vector<int> &axis) {
    if (input.dataDevice != xllm::DataDevice::CUDA) {
        printf("permute: data should in cuda.\n");
        exit(0);
    }
    int len = input.Count(0);
    float *tempData = (float *)xllmCudaMalloc(len * sizeof(float));
    cudaMemcpy(tempData, input.cudaData, len * sizeof(float), cudaMemcpyDeviceToDevice);

    std::vector<int> new_dims;
    for (int i = 0; i < axis.size(); i++) {
        new_dims.push_back(input.dims[axis[i]]);
    }
    if (axis == std::vector <int> {1, 0, 2}) {
        int n = input.dims[0];
        int m = input.dims[1];
        int k = input.dims[2];
        xllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.cudaData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else if (axis == std::vector <int> {2, 0, 1, 3}) {
        int n = input.dims[0] * input.dims[1];
        int m = input.dims[2];
        int k = input.dims[3];
        xllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.cudaData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else {
        std::vector<int> temp;
        int len = input.Count(0);
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(axis[i]);
        }
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(input.Count(i + 1));
        }
        input.Resize(new_dims);
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(input.Count(i + 1));
        }

        int *cudaTemp = (int *) xllmCudaMalloc(temp.size() * sizeof(int));
        cudaMemcpy(cudaTemp, temp.data(), temp.size() * sizeof(int), cudaMemcpyHostToDevice);
        int threadPerBlock = std::min(256, len);
        xllmPermuteKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>((float *) input.cudaData,
                                                                                    tempData, cudaTemp,
                                                                                    (int) axis.size(), len);
        xllmCudaFree(cudaTemp);
    }

    xllmCudaFree(tempData);
    return true;
}

bool xllmCudaBatchMatMulTransB(const xllm::Data &input0, const xllm::Data &input1, xllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha) {
    float *cudaInput0 = (float *) xllmCudaPrepareInput(input0);
    float *cudaInput1 = (float *) xllmCudaPrepareInput(input1);
    float *cudaOutput = (float *) xllmCudaPrepareOutput(output);
    float beta = 0;
    auto xllmCublasHandle = getxllmCublasHandle();
    cublasStatus_t status;

    status = cublasSgemmStridedBatched(xllmCublasHandle,
                                       CUBLAS_OP_T, CUBLAS_OP_N,
                                       k, n, m, &alpha,
                                       cudaInput1, input1Stride, input1Spatial,
                                       cudaInput0, input0Stride, input0Spatial,
                                       &beta,
                                       cudaOutput, k, k * n, batch);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error.\n");
        throw("cublas error");
        exit(0);
    }

    xllmCudaFinishInput(input0, cudaInput0);
    xllmCudaFinishInput(input1, cudaInput1);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}

template <int THREAD_PER_BLOCK>
__global__ void xllmAttentionMaskKernel(float* a, float *b, float maskValue, int n, int m, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        if (b[on * spatial + i] > 0.99) {
            a[o * spatial + i] = maskValue;
        }
    }
}

bool xllmCudaAttentionMask(xllm::Data &input, const xllm::Data &mask, float maskValue) {
    int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];
    float *cudaData = (float *) xllmCudaPrepareInput(input);
    float *maskData = (float *) xllmCudaPrepareInput(mask);

    xllmAttentionMaskKernel <256> <<< n * m, 256>>>(cudaData, maskData, maskValue,
                                                       n, m, spatial);
    xllmCudaFinishInput(mask, maskData);
    xllmCudaFinishOutput(input, cudaData);
    return true;
}

// callable from the device
template <int THREAD_PER_BLOCK>
__device__ void xllmSoftmaxKernelInner1Func(float *input, float *output, int channels) {
    // 共享内存类型(__shared__), 同一个block中的线程之间可以共享
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float maxV;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    unsigned int per = (channels / THREAD_PER_BLOCK);  // 每个线程计算几个数据
    unsigned int id = threadIdx.x * per;
    unsigned int len = per;
    // 最后一个线程把未能整除的数据一起计算
    if (tid == blockDim.x - 1) {
        len += (channels - per * THREAD_PER_BLOCK);
    }
    float maxValue = input[id];
    for (int i = 0; i < len; i++) {
        maxValue = max(maxValue, input[id + i]);
    }
    sdata[tid] = maxValue;
    __syncthreads();

    // 2. 求max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 3. 记录max
    if (tid == 0) {
        maxV = sdata[0];
    }
    __syncthreads();

    // 4. 求和
    float sum = 0;
    for (int i = 0; i < len; i++) {
        output[id + i] = exp(input[id + i] - maxV);
        sum += output[id + i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (fabs(sdata[0]) < 1e-6) {
            sdata[0] = 0.1;
        }
    }
    __syncthreads();

    for (int i = 0; i < len; i++) {
        output[id + i] /= sdata[0];
    }
}

// callable from the host
// 一个block处理一个outer的数据
template <int THREAD_PER_BLOCK>
__global__ void xllmSoftmaxKernelInner1(float* input, float *output, int outer, int channels) {
    int o = blockIdx.x;
    xllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels);
}

bool xllmCudaSoftmax(const xllm::Data &input, xllm::Data &output, int axis) {
    float *cudaInput = (float *) xllmCudaPrepareInput(input);
    float *cudaOutput = (float *) xllmCudaPrepareInput(output);

    // float* hostData = (float*)malloc(input.assignBytes);
    // cudaMemcpy(hostData, cudaInput, input.assignBytes, cudaMemcpyDeviceToHost);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.Count(axis + 1);

    if (inner == 1) {
        if (channels < 8) {
            xllmSoftmaxKernelInner1 <1> <<< outer, 1 >>> (cudaInput, cudaOutput, outer, channels);
        } else if (channels < 64) {
            xllmSoftmaxKernelInner1 <1> <<< outer, 1 >>> (cudaInput, cudaOutput, outer, channels);
        } else if (channels < 512) {
            xllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (cudaInput, cudaOutput, outer, channels);
        } else {
            xllmSoftmaxKernelInner1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, outer, channels);
        }

    } else {
        printf("softmax error.\n");
        exit(0);
    }

    // float* hostDataOut = (float*)malloc(input.assignBytes);
    // cudaMemcpy(hostDataOut, cudaOutput, input.assignBytes, cudaMemcpyDeviceToHost);

    xllmCudaFinishInput(input, cudaInput);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool xllmCudaBatchMatMul(const xllm::Data &input0, const xllm::Data &input1, xllm::Data &output,
                            int input0Spatial, int input1Spatial, int outputSpatial,
                            int input0Stride, int input1Stride,
                            int batch, int n, int m, int k, float alpha) {
    float *cudaInput0 = (float *) xllmCudaPrepareInput(input0);
    float *cudaInput1 = (float *) xllmCudaPrepareInput(input1);
    float *cudaOutput = (float *) xllmCudaPrepareOutput(output);
    float beta = 0;
    auto xllmCublasHandle = getxllmCublasHandle();
    cublasStatus_t status;

    status = cublasSgemmStridedBatched(xllmCublasHandle,
                                       CUBLAS_OP_N, CUBLAS_OP_N,
                                       k, n, m, &alpha,
                                       cudaInput1, input1Stride, input1Spatial,
                                       cudaInput0, input0Stride, input0Spatial,
                                       &beta,
                                       cudaOutput, k, k * n, batch);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error.\n");
        throw("cublas error");
        exit(0);
    }

    xllmCudaFinishInput(input0, cudaInput0);
    xllmCudaFinishInput(input1, cudaInput1);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}

__global__ void xllmAddToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] += b[idx] * alpha;
    }
}

bool xllmCudaAddTo(xllm::Data &input0, const xllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *) xllmCudaPrepareInput(input0);
    float *input1Data = (float *) xllmCudaPrepareInput(input1);

    int threadPerBlock = std::min(256, len);
    xllmAddToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
    xllmCudaFinishInput(input1, input1Data);
    xllmCudaFinishOutput(input0, cudaData);
    return true;
}

__global__ void xllmSiluKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = x / (1.0 + expf(-x));
    }
}

bool xllmCudaSilu(const xllm::Data &input, xllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) xllmCudaPrepareInput(input);
    float *cudaOutput = (float *) xllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);
    xllmSiluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    xllmCudaFinishInput(input, cudaInput);
    xllmCudaFinishOutput(output, cudaOutput);
    return true;
}


__global__ void xllmMulToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] *= b[idx] * alpha;
    }
}

bool xllmCudaMulTo(xllm::Data &input0, const xllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *) xllmCudaPrepareInput(input0);
    float *input1Data = (float *) xllmCudaPrepareInput(input1);

    int threadPerBlock = std::min(256, len);
    xllmMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
    xllmCudaFinishInput(input1, input1Data);
    xllmCudaFinishOutput(input0, cudaData);
    return true;
}
