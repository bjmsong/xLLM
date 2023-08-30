#pragma once

#include "data.h"


namespace xllm{

    typedef std::map <std::string, float> FloatDict;
    typedef std::map <std::string, int> IntDict;
    
    void Embedding(const Data &input, Data &weight, Data &output);

    void RMSNorm(const Data &input, Data &weight, Data &output,float eps);

    void Linear(const Data &input, Data &weight, Data &output);

    void LlamaRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim); // 2D position for llama

    void PermuteSelf(Data &input, Data &axisData);
}
