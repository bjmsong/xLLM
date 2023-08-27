#pragma once

#include <string>

namespace xllm{
    static void ErrorInXLLM(const std::string &error) {
        printf("XLLM Error: %s\n", error.c_str());
        throw error;
    }

    static void AssertInXLLM(bool condition, const std::string &error) {
        if (!condition) {
            ErrorInXLLM(error);
        }
    }
}