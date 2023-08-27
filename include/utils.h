#pragma once

#include <string>

namespace xllm{
    static void ErrorInFastLLM(const std::string &error) {
        printf("FastLLM Error: %s\n", error.c_str());
        throw error;
    }

    static void AssertInFastLLM(bool condition, const std::string &error) {
        if (!condition) {
            ErrorInFastLLM(error);
        }
    }
}