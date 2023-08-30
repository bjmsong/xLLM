#include <gtest/gtest.h>
#include "data.h"
#include "operator.h"
#include "param.h"

using namespace xllm;

WeightMap weight{"/root/autodl-tmp/llama2_7b_chat.bin"};

TEST(test_opeartor, test_opeartor1) {
    Data input = Data(DataType::FLOAT32, {1, 5}, {1,2,3,4,5});
    Data hiddenStates(DataType::FLOAT32, {input.dims[1], 4096});
    Embedding(input, weight["embed_tokens.weight"], hiddenStates);
    ASSERT_EQ(hiddenStates.counts, 5*4096);
    ASSERT_NE(((float *)hiddenStates.cpuData)[0], 0);

    Data attenInput(DataType::FLOAT32, hiddenStates.dims);
    RMSNorm(hiddenStates, weight["layers." + std::to_string(0) + ".input_layernorm.weight"],
        attenInput, 1e-6);

    std::string qWeightName = "layers." + std::to_string(0) + ".self_attn.q_proj.weight";
    std::string kWeightName = "layers." + std::to_string(0) + ".self_attn.k_proj.weight";
    std::string vWeightName = "layers." + std::to_string(0) + ".self_attn.v_proj.weight";
    Data q(DataType::FLOAT32, hiddenStates.dims), k(DataType::FLOAT32, hiddenStates.dims), v(DataType::FLOAT32, hiddenStates.dims);
    Linear(attenInput, weight[qWeightName], q);

}