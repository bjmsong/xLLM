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

    void Linear(const Data &input, Data &weight, Data &output, const FloatDict &floatParams, const IntDict &intParams) {
auto st = std::chrono::system_clock::now();
        Data &bias = *(datas.find("bias")->second);

        output.Allocate(0.0f);
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
                    futures.push_back(pool->Submit(FloatLinearPart, inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }

                FloatLinearPart(inputData, weightData, biasData, outputData, n, m, k, cur, k);
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
                    futures.push_back(pool->Submit(Float16LinearPart, inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }

                Float16LinearPart(inputData, weightData, biasData, outputData, n, m, k, cur, k);
                for (int i = 0; i < futures.size(); i++) {
                    futures[i].get();
                }
            } else if (weight.dataType == DataType::INT8) {
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                weight.CalcWeightSum();

                std::vector<LowBitConfig> inputConfigs;
                for (int i = 0; i < n; i++) {
                    float minValue = 1e9, maxValue = -1e9;
                    for (int j = 0; j < m; j++) {
                        minValue = std::min(minValue, inputData[i * m + j]);
                        maxValue = std::max(maxValue, inputData[i * m + j]);
                    }
                    inputConfigs.push_back(LowBitConfig(minValue, maxValue, 8, 0));
                }
                std::vector<uint8_t> uinput;
                uinput.resize(n * m);
                for (int i = 0; i < n * m; i++) {
#ifdef __AVX2__
                    uinput[i] = inputConfigs[i / m].quantization(inputData[i]);
                    uinput[i] = (uinput[i] + !uinput[i]) ^ 128;
#else
                    uinput[i] = inputConfigs[i / m].quantization(inputData[i]);
#endif
                }

                MultiplyMultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k, GetThreads());
                for (int i = 0; i < n; i++) {
                    uint32_t inputSum = 0;
                    for (int j = 0; j < m; j++) {
#ifdef __AVX2__
                        inputSum += uinput[i * m + j] ^ 128;
#else
                        inputSum += uinput[i * m + j];
#endif
                    }

                    for (int j = 0; j < k; j++) {
                        int value = ((int32_t *) outputData)[i * k + j];
#ifdef __AVX2__
                        value += (128 * weight.weightSum[j]);
                        value += (128 * inputSum);
                        value -= m * 128 * 128;
#endif
                        value -= weight.weightSum[j] * inputConfigs[i].zeroPoint;
                        value -= inputSum * weight.perChannelsConfigs[j].zeroPoint;
                        value += (int) inputConfigs[i].zeroPoint * weight.perChannelsConfigs[j].zeroPoint * m;
                        outputData[i * k + j] = weight.perChannelsConfigs[j].scale * inputConfigs[i].scale * value +
                                                (biasData == nullptr ? 0.0 : biasData[j]);
                    }
                }

                /*
                这部分是float输入，float输出
                int threadNum = threads;
                int per = k / threadNum;
                int cur = 0;
                std::vector<std::thread *> threads;
                for (int i = 0; i < threadNum - 1; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    threads.push_back(new std::thread(&Int8LinearPart, inputData, weightData, biasData, outputData,
                                                      weight.perChannelsConfigs.data(), n, m, k, cur, end));
                    cur = end;
                }
                Int8LinearPart(inputData, weightData, biasData, outputData, weight.perChannelsConfigs.data(), n, m, k, cur, k);
                for (int i = 0; i < threadNum - 1; i++) {
                    threads[i]->join();
                    delete threads[i];
                }
                */
            } else if (weight.dataType == DataType::INT4 || weight.dataType == DataType::INT4_NOZERO) {
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                weight.CalcWeightSum();

                std::vector<LowBitConfig> inputConfigs;
                for (int i = 0; i < n; i++) {
                    float minValue = 1e9, maxValue = -1e9;
                    for (int j = 0; j < m; j++) {
                        minValue = std::min(minValue, inputData[i * m + j]);
                        maxValue = std::max(maxValue, inputData[i * m + j]);
                    }
                    inputConfigs.push_back(LowBitConfig(minValue, maxValue, 8, 0));
                }
                std::vector<uint8_t> uinput;
                uinput.resize(n * m);
                for (int i = 0; i < n * m; i++) {
                    uinput[i] = inputConfigs[i / m].quantization(inputData[i]);
                }
#ifdef __AVX__
                uint8_t *temp = new uint8_t[32];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j + 31 < m; j += 32) {
                        memcpy(temp, uinput.data() + i * m + j, 32);
                        for (int k = 0; k < 16; k++) {
                            uinput[i * m + j + k] = temp[k * 2 + 1];
                            uinput[i * m + j + k + 16] = temp[k * 2];
                        }
                    }
                }
                delete[] temp;
#endif
                if (weight.dataType == DataType::INT4) {
                    MultiplyInt4MultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k,
                                            weight.weightSum.data(), weight.zeros.data(), weight.scales.data(),
                                            biasData,
                                            inputConfigs, GetThreads());
                } else {
                    MultiplyInt4NoZeroMultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k,
                                                  weight.weightSum.data(), weight.mins.data(), weight.scales.data(),
                                                  biasData,
                                                  inputConfigs, GetThreads());
                }

/*
            //这部分是float输入，float输出
            int threadNum = GetThreads();
            int per = k / threadNum;
            int cur = 0;
            std::vector<std::thread *> threads;
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < k);
                threads.push_back(new std::thread(&Int4LinearPart, inputData, weightData, biasData, outputData,
                                                  weight.perChannelsConfigs.data(), n, m, k, cur, end));
                cur = end;
            }
            Int4LinearPart(inputData, weightData, biasData, outputData, weight.perChannelsConfigs.data(), n, m, k, cur, k);
            for (int i = 0; i < threadNum - 1; i++) {
                threads[i]->join();
                delete threads[i];
            }
*/
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
                    futures.push_back(pool->Submit(Float16xFloat16LinearPart, inputData, weightData, biasData, outputData,
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
            ErrorInXLLM("Linear error: unsupport weight's dataType.\n");
        }
float spend = GetSpan(st, std::chrono::system_clock::now());
float gops = (float)n * m * k / spend / 1e9;
printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
    }
}
