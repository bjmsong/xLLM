#include "operator.h"

namespace xllm{
    void Embedding(const Data &input, Data &weight, Data &output) {
        output.Allocate();
        int vocabSize = weight.dims[0], embSize = weight.dims[1];
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


    // input(n,m) * weight(m,k) = output(n,k)
    void Linear(const Data &input, Data &weight, Data &output) {
        output.Allocate();
        auto st = std::chrono::system_clock::now();
        Data bias;

        int n = input.counts / input.dims.back();
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

        float spend = GetSpan(st, std::chrono::system_clock::now());
        float gops = (float)2* n * m * k / spend / 1e9;
        printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
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

    void Permute(Data &input, Data &output, Data &axisData) {
        std::vector <int> axis;
        for (int i = 0; i < axisData.counts; i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }

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

    void PermuteSelf(Data &input, Data &axisData) {
        std::vector <int> axis;
        for (int i = 0; i < axisData.counts; i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }

        AssertInXLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, "Permute error: datatype should be float32 or float16.");
        AssertInXLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");

        bool same = false;
        // 下面几种情况不需要移动数据
        // 例如: (1,2,3) -> (2,3,1) / (2,1,3)
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

        auto tmp = new Data();
        Permute(input, *tmp, axisData);

        memcpy(input.cpuData, tmp->cpuData, input.unitSize * input.counts);
        input.Resize(tmp->dims);
        delete tmp;
    }

}
