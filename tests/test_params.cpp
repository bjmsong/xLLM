#include <gtest/gtest.h>
#include "param.h"


using namespace xllm;

TEST(test_tokenizer, Encode1) {
    Tokenizer tokenizer("/root/autodl-tmp/tokenizer.bin");
    Data data = tokenizer.Encode("hello world!");
    ASSERT_EQ(data.counts, 3);
    ASSERT_EQ(((float*)data.cpuData)[0], 12199);
    ASSERT_EQ(((float*)data.cpuData)[1], 3186);
    // ASSERT_EQ(tokens[2], 36);
}

TEST(test_tokenizer, Encode2) {
    Tokenizer tokenizer("/root/autodl-tmp/tokenizer.bin");
    Data data = tokenizer.Encode("what is the recipe of mayonnaise?");
    ASSERT_EQ(data.counts, 10);
    ASSERT_EQ(((float*)data.cpuData)[0], 5816);
    ASSERT_EQ(((float*)data.cpuData)[1], 338);
    ASSERT_EQ(((float*)data.cpuData)[2], 278);
    ASSERT_EQ(((float*)data.cpuData)[3], 9522);
    ASSERT_EQ(((float*)data.cpuData)[4], 412);
    ASSERT_EQ(((float*)data.cpuData)[5], 310);
    ASSERT_EQ(((float*)data.cpuData)[6], 1122);
    ASSERT_EQ(((float*)data.cpuData)[7], 11586);
    ASSERT_EQ(((float*)data.cpuData)[8], 895);
    // ASSERT_EQ(tokens[9], 66);
}


TEST(test_weightmap, LoadFromFile) {
    WeightMap weightmap("/root/autodl-tmp/llama2_7b_chat.bin");
}