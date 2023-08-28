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

    static double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
        return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    }

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\n\r");
        size_t end = s.find_last_not_of(" \t\n\r");
        
        if (start == std::string::npos) {
            return ""; // 字符串全为空格
        }

        return s.substr(start, end - start + 1);
    }
}