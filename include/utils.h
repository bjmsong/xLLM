#pragma once

#include <string>
#include <chrono>

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

    double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
        return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    };
}