#pragma once

#include "data.h"


namespace xllm{

    typedef std::map <std::string, float> FloatDict;
    typedef std::map <std::string, int> IntDict;
    
    void Embedding(const Data &input, Data &weight, Data &output);

    void RMSNorm(const Data &input, Data &weight, Data &output,float eps);

    void Linear(const Data &input, Data &weight, Data &output);
}
