#include <cfloat>

#include "operator.h"
#include "data.h"

namespace xllm{
    void Embedding(const Data &input, Data &weight, Data &output) {
        output.Allocate();
        int embSize = weight.dims[1];
        float *inputData = (float*)input.cpuData;
        if (weight.dataType == DataType::FLOAT32) {
            float *outputData = (float *) output.cpuData;
            float *weightData = (float *) weight.cpuData;
            for (int i = 0; i < input.counts; i++) {
                int token = (int) (inputData[i] + 1e-9);
                memcpy(outputData + i * embSize, weightData + token * embSize, embSize * sizeof(float));
            }
        } 
    }

    void RMSNorm(const Data &input, Data &weight, Data &output, float eps) {
        output.Allocate();
        int inner = input.dims.back();
        int outer = input.counts / inner;

        if (input.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *outputData = (float *) output.cpuData;
            float *weightData = (float *) weight.cpuData;

            for (int i = 0; i < outer; i++) {
                float mean = 0.f;
                int j = 0;
                for (; j < inner; j++) {
                    mean += inputData[j] * inputData[j];
                }
                float scale = 1.0 / sqrt(mean / inner + eps);
                j = 0;
                for (; j < inner; j++) {
                    outputData[j] = inputData[j] * scale * weightData[j];
                }

                inputData += inner;
                outputData += inner;
            }
        } 
        // else if (input.dataType == DataType::FLOAT16) {
        //     uint16_t *inputData = (uint16_t *) input.cpuData;
        //     uint16_t *outputData = (uint16_t *) output.cpuData;
        //     float *weightData = (float *) weight.cpuData;

        //     for (int i = 0; i < outer; i++) {
        //         float mean = 0.f;
        //         int j = 0;
        //         for (; j < inner; j++) {
        //             float x = fp16tofp32.dict[inputData[j]];
        //             mean += x * x;
        //         }
        //         float scale = 1.0 / sqrt(mean / inner + eps);
        //         j = 0;
        //         for (; j < inner; j++) {
        //             outputData[j] = float_to_half(fp16tofp32.dict[inputData[j]] * scale * weightData[j]);
        //         }

        //         inputData += inner;
        //         outputData += inner;
        //     }
        // } 
        else {
            ErrorInXLLM("RMSNorm error: unsupport dataType.\n");
        }
    }

    // inputData(n, m) * weightData(m, end-st) = outputData(n, end-st)
    void FloatLinearPart(float *inputData, float *weightData, float *biasData, float *outputData,
                            int n, int m, int k, int st, int end) {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    int l = 0;
    #ifdef __AVX2__
                    __m256 vsum = _mm256_setzero_ps();
                    for (; l + 7 < m; l += 8) {
                        __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                        __m256 vw = _mm256_loadu_ps(weightData + j * m + l);
                        vsum = _mm256_fmadd_ps(vi, vw, vsum);
                    }
                    now += Floatsum(vsum);
    #endif
                    for (; l < m; l++) {
                        now += inputData[i * m + l] * weightData[j * m + l];
                    }
                    outputData[i * k + j] = now;
                }
            }
        }

    // uint16_t -> float32
    struct FP16ToFP32Manager {
        float dict[65536];  // 2^16

        FP16ToFP32Manager() {
            for (uint16_t i = 0; i < 65535; i++) {
                dict[i] = half_to_float(i);
            }
        }
    } fp16tofp32;

    // inputData(float32) * weightData(float16) = outputData(float32)
    void Float16LinearPart(float *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
#ifdef __AVX2__
                __m256 vsum = _mm256_setzero_ps();
                for (; l + 7 < m; l += 8) {
                    __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                    // _mm256_cvtph_ps: float16 -> float32 
                    __m256 vw = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (weightData + j * m + l)));
                    vsum = _mm256_fmadd_ps(vi, vw, vsum);
                }
                now += Floatsum(vsum);
#endif
                for (; l < m; l++) {
                    now += inputData[i * m + l] * fp16tofp32.dict[weightData[j * m + l]];
                }
                outputData[i * k + j] = now;
            }
        }
    }

    // inputData(float16) * weightData(float16) = outputData(float16)
    void Float16xFloat16LinearPart(uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData,
                           int n, int m, int k, int st, int end) {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
                for (; l < m; l++) {
                    // TODO: SIMD加速
                    now += inputData[i * m + l] * fp16tofp32.dict[weightData[j * m + l]];
                }
                outputData[i * k + j] = float_to_half(now);
            }
        }
    }


    // 多维矩阵*二维矩阵乘法：input(n,m) * weight(k,m) + bias = output(n,k)
    void Linear(const Data &input, Data &weight, Data &output) {
        output.Allocate(0);
        // auto st = std::chrono::system_clock::now();
        Data bias;

        int n = input.counts / input.dims.back();  // 前面的维度打包到一个维度
        int m = input.dims.back();
        int k = output.dims.back();

        if (input.dataType == DataType::FLOAT32 && output.dataType == DataType::FLOAT32) {
            if (weight.dataType == DataType::FLOAT32) {
                float *inputData = (float *) input.cpuData;
                float *weightData = (float *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;

                int threadNum = GetThreads();
                int per = k / threadNum;
                int cur = 0;
                auto pool = GetPool();
                std::vector<std::future<void> > futures;
                for (int i = 0; i < threadNum - 1; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    futures.push_back(pool->enqueue(FloatLinearPart, inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }

                FloatLinearPart(inputData, weightData, biasData, outputData, n, m, k, cur, k);  // 如果k不能被threadNum整除
                for (int i = 0; i < futures.size(); i++) {
                    futures[i].get();
                }
            } else if (weight.dataType == DataType::FLOAT16) {
                float *inputData = (float *) input.cpuData;
                // 大部分CPU不支持FP16类型，但是支持uint16_t
                uint16_t *weightData = (uint16_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                int threadNum = GetThreads();
                int per = k / threadNum;
                int cur = 0;
                auto pool = GetPool();
                std::vector<std::future<void> > futures;
                for (int i = 0; i < threadNum - 1; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    futures.push_back(pool->enqueue(Float16LinearPart, inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }

                Float16LinearPart(inputData, weightData, biasData, outputData, n, m, k, cur, k);
                for (int i = 0; i < futures.size(); i++) {
                    futures[i].get();
                }
            } else {
                ErrorInXLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else if (input.dataType == DataType::FLOAT16 && output.dataType == DataType::FLOAT16) {
            if (weight.dataType == DataType::FLOAT16) {
                uint16_t *inputData = (uint16_t *) input.cpuData;
                uint16_t *weightData = (uint16_t *) weight.cpuData;
                uint16_t *outputData = (uint16_t *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                int threadNum = GetThreads();
                int per = k / threadNum;
                int cur = 0;
                auto pool = GetPool();
                std::vector<std::future<void> > futures;
                for (int i = 0; i < threadNum - 1; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    futures.push_back(pool->enqueue(Float16xFloat16LinearPart, inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }

                Float16xFloat16LinearPart(inputData, weightData, biasData, outputData, n, m, k, cur, k);
                for (int i = 0; i < futures.size(); i++) {
                    futures[i].get();
                }
            } else {
                ErrorInXLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else {
            ErrorInXLLM("Linear error: unsupport input's dataType.\n");
        }

        // float spend = GetSpan(st, std::chrono::system_clock::now());
        // float gops = (float)2* n * m * k / spend / 1e9;
        // printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
    }

    void LlamaRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim) {

        int bsz = input.dims[0], seqlen = input.dims[1];
        int spatial = input.Count(2);
        int n = input.dims[2], m = input.dims[3];
        int stride = (int)sinData.dims[1];
        for (int b = 0; b < bsz; b++) {
            for (int l = 0; l < seqlen; l++) {
                int index = (int) ((float *) positionIds.cpuData)[b * positionIds.dims.back() + l];
                float *sin = ((float *) sinData.cpuData) + stride * index;
                float *cos = ((float *) cosData.cpuData) + stride * index;
                float *d = (float *) input.cpuData + (b * seqlen + l) * spatial;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < rotaryDim && j < m / 2; j++) {
                        float a = d[j], b = d[j + m / 2];
                        d[j] = a * cos[j] - b * sin[j];
                        d[j + m / 2] = a * sin[j] + b * cos[j];
                    }

                    d += m;
                }
            }
        }
    }

    void Transpose4x4(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
        if (n < 4 || m < 4) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    pDst[j * dstStride + i] = pSrc[i * srcStride + j];
                }
            }

            return;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                pDst[j * dstStride + i] = pSrc[i * srcStride + j];
            }
        }
    }

    void Transpose(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
        int per = 4;
        for (int i = 0; i < n; i += per) {
            for (int j = 0; j < m; j += per) {
                Transpose4x4(pDst + j * dstStride + i,
                             pSrc + i * srcStride + j,
                             dstStride, srcStride,
                             std::min(per, n - i),
                             std::min(per, m - j));
            }
        }
    }

    void Permute(Data &input, Data &output, std::vector <int> axis) {

        output.Allocate();
        uint8_t *tmpData = (uint8_t *) output.cpuData;
        uint8_t *curData = (uint8_t *) input.cpuData;

        if (axis == std::vector <int> {1, 2, 0} && input.dataType == DataType::FLOAT32) {
            int n = input.dims[0];
            int m = input.Count(1);

            int threadNum = 1;
            int per = m / threadNum;
            int cur = 0;
            auto pool = GetPool();
            std::vector <std::future <void> > futures;
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < m);
                futures.push_back(pool->enqueue(Transpose, ((float*)tmpData) + cur * n, ((float*)curData) + cur, n, m, n, end - cur));
                cur = end;
            }
            Transpose(((float*)tmpData) + cur * n, ((float*)curData) + cur, n, m, n, m - cur);
            for (int i = 0; i < futures.size(); i++) {
                futures[i].get();
            }
        } else if (axis == std::vector <int> {1, 0, 2}) {
            int n = input.dims[0];
            int m = input.dims[1];
            int k = input.dims[2];
            int unitSize = input.unitSize;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    memcpy(tmpData + (j * n + i) * k * unitSize, curData + (i * m + j) * k * unitSize, k * unitSize);
                }
            }
        } else if (axis == std::vector <int> {2, 0, 1, 3}) {
            int n = input.dims[0] * input.dims[1];
            int m = input.dims[2];
            int k = input.dims[3];
            int unitSize = input.unitSize;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    memcpy(tmpData + (j * n + i) * k * unitSize, curData + (i * m + j) * k * unitSize, k * unitSize);
                }
            }
        } else if (axis == std::vector<int> {0, 2, 1, 3}) {
            int b = input.dims[0];
            int n = input.dims[1];
            int m = input.dims[2];
            int k = input.dims[3];
            int unitSize = input.unitSize;
            for (int o = 0; o < b; o++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        memcpy(tmpData + (j * n + i) * k * unitSize, curData + (i * m + j) * k * unitSize, k * unitSize);
                    }
                }
                tmpData += output.Count(1) * unitSize;
                curData += input.Count(1) * unitSize;
            }
        } else {
            std::vector<int> oldSteps;
            std::vector<int> newSteps;
            int count = input.Count(0);
            auto oldPos = new int[count];
            for (int i = 0; i < axis.size(); i++) {
                oldSteps.push_back(input.Count(i + 1));
                newSteps.push_back(output.Count(i + 1));
            }

            for (int i = 0; i < count; ++i) {
                int old = 0;
                int idx = i;
                for (int j = 0; j < axis.size(); ++j) {
                    int order = axis[j];
                    old += (idx / newSteps[j]) * oldSteps[order];
                    idx %= newSteps[j];
                }
                oldPos[i] = old;
            }

            if (input.unitSize == 4) {
                for (int i = 0; i < count; ++i) {
                    ((float*)tmpData)[i] = ((float*)curData)[oldPos[i]];
                }
            } else if (input.unitSize == 2) {
                for (int i = 0; i < count; ++i) {
                    ((uint16_t*)tmpData)[i] = ((uint16_t*)curData)[oldPos[i]];
                }
            } else if (input.unitSize == 1) {
                for (int i = 0; i < count; ++i) {
                    ((uint8_t*)tmpData)[i] = ((uint8_t*)curData)[oldPos[i]];
                }
            }

            delete[] oldPos;
        }
    }

    void PermuteSelf(Data &input, std::vector <int> axis) {

        AssertInXLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, "Permute error: datatype should be float32 or float16.");
        AssertInXLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");

        std::vector<int> new_dims;
        for (int i = 0; i < axis.size(); i++) {
            new_dims.push_back(input.dims[axis[i]]);
        }
        
        bool same = false;
        // 下面几种情况不需要移动数据
        // 例如: (1,2,3) -> (2,3,1) / (2,1,3)
        same |= ((axis == std::vector <int>{1, 2, 0} || axis == std::vector <int>{1, 0, 2}) && (input.dims[0] == 1 || input.dims[1] == 1));
        same |= ((axis == std::vector <int>{2, 0, 1, 3}) && input.dims[2] == 1);
        same |= ((axis == std::vector <int>{0, 2, 1, 3}) && (input.dims[1] == 1 || input.dims[2] == 1));
        if (same) {
            input.Resize(new_dims);
            return;
        }

        auto tmp = new Data(DataType::FLOAT32, new_dims);
        Permute(input, *tmp, axis);

        memcpy(input.cpuData, tmp->cpuData, input.unitSize * input.counts);
        input.Resize(tmp->dims);
        delete tmp;
    }

    void CatDirect(Data &input0, Data &input1, int axis) {

        AssertInXLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                        "CatDirect's input's type should be float32.\n");

        // input0还没有数据
        if (input0.dims.size() == 0) {
            input0.counts = input1.counts;
            input0.bytes = input1.bytes;
            input0.Resize(input1.dims);

            AssertInXLLM(input0.expandDims.size() == input1.dims.size() &&
                            input1.dims[axis] <= input0.expandDims[axis],
                            "CatDirect Error: input0's expansion size is not enough.\n");

            int outer = input1.counts / input1.Count(axis);
            int input1Stride = input1.Count(axis);
            int input0Stride = input1Stride/input1.dims[axis] * input0.expandDims[axis];
            int unitSize = input0.unitSize;
            for (int o = 0; o < outer; o++) {
                memcpy(input0.cpuData + o * input0Stride * unitSize,
                       input1.cpuData + o * input1Stride * unitSize,
                       input1Stride * unitSize);
            }

            return;
        }

        // input0已有数据
        AssertInXLLM(input0.dims.size() == input1.dims.size(), "Cat Error: input's shape's size should be same.\n");
        int dimsLen = input0.dims.size();
        // axis = (axis % dimsLen + dimsLen) % dimsLen;

        for (int i = 0; i < dimsLen; i++) {
            if (i != axis) {
                AssertInXLLM(input0.dims[i] == input1.dims[i], "Cat Error: input's shape doesn't match.");
            }
        }

        std::vector <int> dims = input0.dims;
        std::vector <int> oldDims = dims;
        dims[axis] += input1.dims[axis];
        input0.Resize(dims);
        input0.counts += input1.counts;
        int unitSize = input0.unitSize;
        input0.bytes += (input1.Count(axis) * unitSize - 1) / input0.unitSizeDiv + 1;

        int outer = input0.counts / input0.Count(axis);
        int input1Stride = input1.Count(axis);
        int input0Stride = input1Stride/input1.dims[axis] * input0.expandDims[axis];
        int inner = input0.strides[axis];
        for (int o = 0; o < outer; o++) {
            memcpy(input0.cpuData + o * input0Stride * unitSize + oldDims[axis] * inner * unitSize,
                   input1.cpuData + o * input1Stride * unitSize,
                   input1Stride * unitSize);
        }
    }

    void MatMulTransBSingle(float *input0Base, float *input1Base, float *outputBase,
                                int input0Spatial, int input1Spatial, int outputSpatial,
                                int input0Stride, int input1Stride,
                                int n, int m, int k, float alpha, int st, int end) {
            for (int b = st; b < end; b++) {
                float *input0Data = input0Base + b * input0Spatial;
                float *input1Data = input1Base + b * input1Spatial;
                float *outputData = outputBase + b * outputSpatial;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < k; j++) {
                        float now = 0.0f;
                        int l = 0;
    #if defined(__AVX__)
                        __m256 vsum = _mm256_set1_ps(0.0f);
                        for (; l + 7 < m; l += 8) {
                            __m256 vx = _mm256_loadu_ps((const float *) (input0Data + i * input0Stride + l));
                            __m256 vy = _mm256_loadu_ps((const float *) (input1Data + j * input1Stride + l));
                            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vy));
                        }
                        now += Floatsum(vsum);
    #endif
                        for (; l < m; l++) {
                            now += input0Data[i * input0Stride + l] * input1Data[j * input1Stride + l];
                        }
                        outputData[i * k + j] = now * alpha;
                    }
                }
            }
        }

    // 三维矩阵乘法: input0(batch, n, m) * input1(batch, k, m)^T = output(batch, n, k)
    // TODO: 可以跟Linear整合吗？
    void MatMulTransB(Data &input0, Data &input1, Data &output, float alpha) {
        output.Allocate();

        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.dims.back()*input1.expandDims[input1.dims.size() - 2];
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int outputSpatial = output.Count(output.dims.size() - 2);

        int batch = input0.counts / input0Spatial;
        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 2];

        int threadNum = GetThreads();
        if (batch * n * m * k < 64 * 4096) {
            threadNum = 1;
        }
        threadNum = std::min(threadNum, 4);
        int per = batch / threadNum;
        int cur = 0;
        auto pool = GetPool();
        std::vector <std::future <void> > futures;
        if (input0.dataType == DataType::FLOAT32) {
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < batch);
                futures.push_back(pool->enqueue(MatMulTransBSingle,
                                               (float *) input0.cpuData, (float *) input1.cpuData,
                                               (float *) output.cpuData,
                                               input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                                               n, m, k, alpha, cur, end));
                cur = end;
            }
            MatMulTransBSingle((float *) input0.cpuData, (float *) input1.cpuData, (float *) output.cpuData,
                               input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                               n, m, k, alpha, cur, batch);
        } 
        for (int i = 0; i < futures.size(); i++) {
            futures[i].get();
        }
    }


    void AttentionMask(Data &input, const Data &mask, float maskValue) {
        int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];

        AssertInXLLM(mask.dataType == DataType::FLOAT32, "AttentionMask: mask's datatype should be float32.");
        if (input.dataType == DataType::FLOAT32) {
            float *maskData = (float *) mask.cpuData;
            float *attnData = (float *) input.cpuData;
            for (int on = 0; on < n; on++) {
                for (int om = 0; om < m; om++) {
                    int o = on * m + om;
                    for (int i = 0; i < spatial; i++) {
                        if (maskData[on * spatial + i] > 0.99) {
                            attnData[o * spatial + i] = maskValue;
                        }
                    }
                }
            }
        } else if (input.dataType == DataType::FLOAT16) {
            float *maskData = (float *) mask.cpuData;
            uint16_t *attnData = (uint16_t *) input.cpuData;
            uint16_t hMaskValue = float_to_half(maskValue);
            for (int on = 0; on < n; on++) {
                for (int om = 0; om < m; om++) {
                    int o = on * m + om;
                    for (int i = 0; i < spatial; i++) {
                        if (maskData[on * spatial + i] > 0.99) {
                            attnData[o * spatial + i] = hMaskValue;
                        }
                    }
                }
            }
        } else {
            ErrorInXLLM("AttentionMask error: unsupport input's dataType.\n");
        }
    }

    void SoftMax(Data &input, Data &output, int axis) {
        output.Allocate();

        AssertInXLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Softmax error: Data's type should be float32.\n");

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.Count(axis + 1);

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        if (input.dataType == DataType::FLOAT16) {
            int len = input.Count(0);
            inputData = new float[len];
            outputData = new float[len];
            for (int i = 0; i < len; i++) {
                inputData[i] = fp16tofp32.dict[((uint16_t *) input.cpuData)[i]];
            }
        }

        if (inner == 1) {
            for (int i = 0; i < outer; i++) {
                float maxValue = 0;
                int j = 0;
                for (; j < channels; j++) {
                    maxValue = std::max(maxValue, inputData[j]);
                }

                j = 0;
                for (; j < channels; j++) {
                    outputData[j] = exp(inputData[j] - maxValue);
                }
                float sum = 0.0;
                j = 0;
                for (; j < channels; j++) {
                    sum += outputData[j];
                }
                if (fabs(sum) < 1e-9) {
                    sum = 0.1;
                }
                j = 0;
                for (; j < channels; j++) {
                    outputData[j] = outputData[j] / sum;
                }
                inputData += channels;
                outputData += channels;
            }
        } else {
            for (int i = 0; i < outer; i++) {
                std::vector<float> maxValue(inner, -FLT_MAX);
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        maxValue[k] = std::max(maxValue[k], inputData[j * inner + k]);
                    }
                }
                std::vector<float> sum(inner, 0.0);
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        outputData[j * inner + k] = std::exp(inputData[j * inner + k] - maxValue[k]);
                        sum[k] += outputData[j * inner + k];
                    }
                }

                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        outputData[j * inner + k] /= sum[k];
                    }
                }

                inputData += channels * inner;
                outputData += channels * inner;
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            int len = input.Count(0);
            inputData -= len;
            outputData -= len;
            for (int i = 0; i < len; i++) {
                ((uint16_t *) output.cpuData)[i] = float_to_half(outputData[i]);
            }

            delete[] inputData;
            delete[] outputData;
        }
    }

    void MatMulSingle(float *input0Base, float *input1Base, float *outputBase,
                      int input0Spatial, int input1Spatial, int outputSpatial,
                      int input0Stride, int input1Stride,
                      int n, int m, int k, float alpha, int st, int end) {
        for (int b = st; b < end; b++) {
            float *input0Data = input0Base + b * input0Spatial;
            float *input1Data = input1Base + b * input1Spatial;
            float *outputData = outputBase + b * outputSpatial;
            std::fill(outputData, outputData + n * k, 0.0f);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    float now = input0Data[i * input0Stride + j] * alpha;
                    for (int l = 0; l < k; l++) {
                        outputData[i * k + l] += (now * input1Data[j * k + l]);
                    }
                }
            }
        }
    }

    // TODO: 跟Linear()进行整合
    // input0(n, m) * input1(m,k) = output(n,k)
    void MatMul(Data &input0, Data &input1, Data &output) {
        output.Allocate();

        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.dims.back()*input1.expandDims[input1.dims.size() - 2];
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims.back();
        int batch = input0.Count(0) / input0Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2);
        int threadNum = GetThreads();
        if (batch * n * m * k < 64 * 4096) {
            threadNum = 1;
        }
        threadNum = std::min(threadNum, 4);
        // TODO: SIMD优化
        int per = batch / threadNum;
        int cur = 0;
        auto pool = GetPool();
        std::vector <std::future <void> > futures;
        float alpha = 1.0;
        if (input0.dataType == DataType::FLOAT32) {
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < batch);
                futures.push_back(pool->enqueue(MatMulSingle,
                                               (float *) input0.cpuData, (float *) input1.cpuData,
                                               (float *) output.cpuData,
                                               input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                                               n, m, k, alpha, cur, end));
                cur = end;
            }
            MatMulSingle((float *) input0.cpuData, (float *) input1.cpuData, (float *) output.cpuData,
                         input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                         n, m, k, alpha, cur, batch);
        } 
        for (int i = 0; i < futures.size(); i++) {
            futures[i].get();
        }
    }

    void AddTo(Data &input0, Data &input1) {
        float alpha = 1.0;

        AssertInXLLM(input0.dataType == DataType::FLOAT32 || input1.dataType == DataType::FLOAT16,
                        "AddTo error: Data's type should be float32 or float16.\n");
        AssertInXLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");

        int len = input0.Count(0);

        if (input0.dataType == DataType::FLOAT32) {
            float *input0Data = (float *) input0.cpuData;
            float *input1Data = (float *) input1.cpuData;
            for (int i = 0; i < len; i++) {
                input0Data[i] += input1Data[i] * alpha;
            }
        } else if (input0.dataType == DataType::FLOAT16) {
            uint16_t *input0Data = (uint16_t *) input0.cpuData;
            uint16_t *input1Data = (uint16_t *) input1.cpuData;
            for (int i = 0; i < len; i++) {
                input0Data[i] = float_to_half(fp16tofp32.dict[input0Data[i]] + fp16tofp32.dict[input1Data[i]] * alpha);
            }
        }
    }

    void MulTo(Data &input0, Data &input1) {
        AssertInXLLM(input0.dims == input1.dims, "MulTo error: input's shape should be same.\n");

        float *input0Data = (float*)input0.cpuData;
        float *input1Data = (float*)input1.cpuData;

        int len = input0.Count(0);
        int inner = input1.Count(0);
        AssertInXLLM(len % inner == 0, "MulTo error: Data`s shape can`t perform MulTo operation.\n");
        int round = (len / inner);
        for (int j = 0; j < round; j++) {
            for (int i = 0; i < len; i++) {
               input0Data[i] *= input1Data[i];
            }
            input0Data += inner;
        }
    }

    void Silu(Data &input, Data &output) {
        output.Allocate();
        AssertInXLLM(input.dataType == DataType::FLOAT32, "Silu error: Data's type should be float32.\n");
        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        int i = 0;
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = x / (1.0 + expf(-x));
        }
    }

}
