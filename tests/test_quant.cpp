#include <gtest/gtest.h>
#include <stdio.h>
#include "data.h"
#include "operator.h"
#include "param.h"

using namespace xllm;


xllm::WeightMap weight("/root/autodl-tmp/llama2_7b_chat.bin");
xllm::WeightMap weightQuant("/root/autodl-tmp/llama2_7b_chat_int8.bin");

TEST(test_quant, linear) {
    Data& lmHead = weight.weight["lm_head.weight"];
    Data& lmHeadInt8 = weightQuant.weight["lm_head.weight"];

    // Data& lmHead = weight.weight["model.layers.0.self_attn.q_proj.weight"];
    // Data& lmHeadInt8 = weightInt8.weight["model.layers.0.self_attn.q_proj.weight"];

    ASSERT_EQ(lmHead.counts, lmHeadInt8.counts);
    ASSERT_EQ(lmHead.dims[0], lmHeadInt8.dims[0]);
    ASSERT_EQ(lmHead.dims[1], lmHeadInt8.dims[1]);
    ASSERT_EQ(lmHead.dataType, DataType::FLOAT16);
    ASSERT_EQ(lmHeadInt8.dataType, DataType::INT8);

    int row = lmHead.dims[0], column = lmHead.dims[1];
    float maxValue = -9999;
    int maxi=0, maxj = 0;
    for (int i = 0; i < row; i++){
        LowBitConfig lbc = lmHeadInt8.perChannelsConfigs[i];
        for (int j = 0; j < column; j++){
            uint8_t valInt8 = ((uint8_t*)lmHeadInt8.cpuData)[i*column + j];
            float val2 = lbc.invQuantization(valInt8);
            if (val2>maxValue){
                maxValue = val2;
                maxi = i;
                maxj= j;
            }
        }
    }

    float valF = ((float*)lmHead.cpuData)[maxi*column + maxj];

    LowBitConfig lbc = lmHeadInt8.perChannelsConfigs[maxi];
    uint8_t valInt8 = ((uint8_t*)lmHeadInt8.cpuData)[maxi*column + maxj];
    float valI = lbc.invQuantization(valInt8);
    ASSERT_NEAR(valF, valI, 0.001);
}


TEST(test_quant, embedding) {
    std::string weightname = "model.embed_tokens.weight";
    Data& emb = weight.weight[weightname];
    Data& embQuant = weightQuant.weight[weightname];

    ASSERT_EQ(emb.counts, embQuant.counts);
    ASSERT_EQ(emb.dims[0], embQuant.dims[0]);
    ASSERT_EQ(emb.dims[1], embQuant.dims[1]);
    ASSERT_EQ(emb.dataType, DataType::FLOAT32);
    ASSERT_EQ(embQuant.dataType, DataType::BFLOAT16);

    ASSERT_EQ(emb.cpuData[2], embQuant.cpuData[0]);
    ASSERT_EQ(emb.cpuData[3], embQuant.cpuData[1]);
    ASSERT_EQ(emb.cpuData[6], embQuant.cpuData[2]);
    ASSERT_EQ(emb.cpuData[7], embQuant.cpuData[3]);
}