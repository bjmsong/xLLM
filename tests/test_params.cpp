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
    Data data = tokenizer.Encode("hello world!", true);
    ASSERT_EQ(data.counts, 4);
    ASSERT_EQ(((float*)data.cpuData)[0], 1);
    ASSERT_EQ(((float*)data.cpuData)[1], 12199);
    ASSERT_EQ(((float*)data.cpuData)[2], 3186);
    // ASSERT_EQ(tokens[2], 36);
}

TEST(test_tokenizer, Encode3) {
    Tokenizer tokenizer("/root/autodl-tmp/tokenizer.bin");
    Data data = tokenizer.Encode((std::string("[INST] ") + "what is the recipe of mayonnaise?" + " [/INST]"), true);
    ASSERT_EQ(data.counts, 18);
    ASSERT_EQ(((float*)data.cpuData)[0], 1);
    ASSERT_EQ(((float*)data.cpuData)[2], 25580);
    ASSERT_EQ(((float*)data.cpuData)[3], 29962);
    ASSERT_EQ(((float*)data.cpuData)[4], 825);
    ASSERT_EQ(((float*)data.cpuData)[5], 338);
    ASSERT_EQ(((float*)data.cpuData)[6], 278);
    ASSERT_EQ(((float*)data.cpuData)[7], 9522);
    ASSERT_EQ(((float*)data.cpuData)[8], 412);
    ASSERT_EQ(((float*)data.cpuData)[9], 310);
}


TEST(test_weightmap, LoadFromFile) {
    WeightMap weightmap("/root/autodl-tmp/llama2_7b_chat.bin");
}

TEST(test_utils, trim) {
    ASSERT_EQ(trim(" abc"), "abc");
    ASSERT_EQ(trim(" abc "), "abc");
    ASSERT_EQ(trim("abc "), "abc");
    ASSERT_EQ(trim("a bc "), "a bc");
}
