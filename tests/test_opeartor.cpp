#include <gtest/gtest.h>
#include "data.h"
#include "operator.h"
#include "param.h"

using namespace xllm;

WeightMap weightmap{"/root/autodl-tmp/llama2_7b_chat.bin"};

TEST(test_opeartor, Embedding_RMSNorm) {
    Data input = Data(DataType::FLOAT32, {1, 5}, {1,2,3,4,5});
    Data hiddenStates(DataType::FLOAT32, {input.dims[1], 4096});
    Embedding(input, weightmap["embed_tokens.weight"], hiddenStates);
    ASSERT_EQ(hiddenStates.counts, 5*4096);
    ASSERT_NE(((float *)hiddenStates.cpuData)[0], 0);

    Data attenInput(DataType::FLOAT32, hiddenStates.dims);
    RMSNorm(hiddenStates, weightmap["layers." + std::to_string(0) + ".input_layernorm.weight"],
        attenInput, 1e-6);
}