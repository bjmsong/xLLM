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

    struct FP16ToFP32Manager {
        float dict[65536];

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
                    // _mm256_cvtph_ps: 将的16位半精度浮点数（__m128i类型）转换为256位单精度浮点数
                    __m256 vw = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (weightData + j * m + l)));
                    vsum = _mm256_fmadd_ps(vi, vw, vsum);
                }
                now += Floatsum(vsum);
#endif
                for (; l < m; l++) {
                    // float16*float32会有精度丢失，因此需要fp16tofp32
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
}
