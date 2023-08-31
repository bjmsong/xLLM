#include <gtest/gtest.h>
#include "data.h"
#include "operator.h"
#include "param.h"

using namespace xllm;


TEST(test_operator, all) {
    WeightMap weight{"/root/autodl-tmp/llama2_7b_chat.bin"};

    Data tokens = Data(DataType::FLOAT32, {1, 5}, {1,2,3,4,5});
    Data hiddenStates(DataType::FLOAT32, {tokens.dims[1], 4096});
    Embedding(tokens, weight["embed_tokens.weight"], hiddenStates);
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

    ModelArgs params;
    int bsz = 1, seqlen = attenInput.dims[0];
    std::vector <int> qkvSize = {bsz, seqlen, params.num_attention_heads, -1};
    q.Reshape(qkvSize);

    std::vector<std::vector<float> > sin, cos;
    sin.resize(params.max_positions);
    cos.resize(params.max_positions);
    std::vector <float> invFreq;
    for (int i = 0; i < params.rotary_dim; i += 2) {
        invFreq.push_back(1.0 / pow(10000, (float)i / params.rotary_dim));
    }
    for (int i = 0; i < params.max_positions; i++) {
        sin[i].resize(params.rotary_dim);
        cos[i].resize(params.rotary_dim);
        for (int j = 0; j < invFreq.size(); j++) {
            sin[i][j] = ::sin((float)i * invFreq[j]);
            cos[i][j] = ::cos((float)i * invFreq[j]);
        }
    }
    std::vector <float> fsin, fcos;
    for (int i = 0; i < sin.size(); i++) {
        for (int j = 0; j < sin[0].size(); j++) {
            fsin.push_back(sin[i][j]);
            fcos.push_back(cos[i][j]);
        }
    }
    Data sinData, cosData;
    sinData.CopyFrom(Data(DataType::FLOAT32, {(int)sin.size(), (int)sin[0].size()}, fsin));
    cosData.CopyFrom(Data(DataType::FLOAT32, {(int)cos.size(), (int)cos[0].size()}, fcos));

    std::vector <float> vpids = std::vector <float> (seqlen, 0);
    for (int i = 0; i < seqlen; i++) {
        vpids[i] = i;
    }
    Data positionIds = Data(DataType::FLOAT32, {1, seqlen}, vpids);

    LlamaRotatePosition2D(q, positionIds, sinData, cosData, params.rotary_dim);

}

TEST(test_operator, Linear_fp32) {
    Data input(DataType::FLOAT32, {2, 2}, {1,2,3,4});
    Data weight(DataType::FLOAT32, {2, 2}, {1,2,3,4});
    Data output(DataType::FLOAT32, {2,2});
    Linear(input, weight, output);
    ASSERT_EQ(((float*)output.cpuData)[0], 5);
    ASSERT_EQ(((float*)output.cpuData)[1], 11);
    ASSERT_EQ(((float*)output.cpuData)[2], 11);
    ASSERT_EQ(((float*)output.cpuData)[3], 25);
}


TEST(test_operator, PermuteSelf_102) {
    std::vector<float> v;
    for (int i = 0; i < 24; i ++) {
        v.push_back(i);
    }
    Data input(DataType::FLOAT32, {2, 3, 4}, v);
    PermuteSelf(input, {1, 0, 2});
    ASSERT_EQ(((float*)input.cpuData)[0], 0);
    ASSERT_EQ(((float*)input.cpuData)[4], 12);
    ASSERT_EQ(((float*)input.cpuData)[8], 4);
    ASSERT_EQ(input.dims[0], 3);
    ASSERT_EQ(input.dims[1], 2);
    ASSERT_EQ(input.dims[2], 4);
    ASSERT_EQ(input.counts, 24);
}

TEST(test_operator, CatDirect) {
    Data pastKey(DataType::FLOAT32);
    std::vector<float> k_vector;
    for (int i = 0; i < 6; i ++) {
        k_vector.push_back(i);
    }
    Data k(DataType::FLOAT32, {2, 1, 3}, k_vector);
    CatDirect(pastKey, k, 1);
}