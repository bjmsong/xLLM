#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <math.h>
#include "threadpool.h"

namespace xllm {
    void PrintInstructionInfo();
    void SetThreads(int t);
    int GetThreads();
    ThreadPool *GetPool();

    struct GenerationConfig {
        int output_token_limit = -1; // 最多输出多少, <= 0代表无限制
        int last_n = 64; // 末尾last_n个token计入重复惩罚
        float repeat_penalty = 1.0f; // 重复惩罚系数，1.0代表不惩罚
        int top_k = 1; // top_k采样
        float top_p = 1.0; // top_p采样
        float temperature = 1.0; // 温度参数，一般在0.1 ~ 1.0之间，设大这个参数可以带来结果的多样性

        bool IsSimpleGreedy() const {
            if (fabs(repeat_penalty - 1) > 1e-8) {
                return false;
            }
            if (top_k > 1) {
                return false;
            }
            return true;
        }
    };

}