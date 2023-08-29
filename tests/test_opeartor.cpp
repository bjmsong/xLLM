#include <gtest/gtest.h>
#include "data.h"
#include "operator.h"
#include "param.h"

using namespace xllm;

WeightMap weightmap{"/root/autodl-tmp/llama2_7b_chat.bin"};

TEST(test_opeartor, Embedding) {
    Data input = Data(DataType::FLOAT32, {1, 5}, {1,2,3,4,5});
    Data output(DataType::FLOAT32, {input.dims[1], 4096});
    Embedding(input, weightmap["embed_tokens.weight"], output);
    ASSERT_EQ(output.counts, 5*4096);
    ASSERT_NE(((float *)output.cpuData)[0], 0);
}