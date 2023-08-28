#include <gtest/gtest.h>
#include "param.h"

using namespace xllm;

TEST(test_tokenizer, Encode1) {
    Tokenizer tokenizer("/root/autodl-tmp/tokenizer.bin");
    std::vector<int> tokens;
    tokenizer.Encode("hello world!", tokens);
    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0], 12199);
    ASSERT_EQ(tokens[1], 3186);
    // ASSERT_EQ(tokens[2], 36);
}

TEST(test_tokenizer, Encode2) {
    Tokenizer tokenizer("/root/autodl-tmp/tokenizer.bin");
    std::vector<int> tokens;
    tokenizer.Encode("what is the recipe of mayonnaise?", tokens);
    ASSERT_EQ(tokens.size(), 10);
    ASSERT_EQ(tokens[0], 5816);
    ASSERT_EQ(tokens[1], 338);
    ASSERT_EQ(tokens[2], 278);
    ASSERT_EQ(tokens[3], 9522);
    ASSERT_EQ(tokens[4], 412);
    ASSERT_EQ(tokens[5], 310);
    ASSERT_EQ(tokens[6], 1122);
    ASSERT_EQ(tokens[7], 11586);
    ASSERT_EQ(tokens[8], 895);
    // ASSERT_EQ(tokens[9], 66);
}


TEST(test_weightmap, LoadFromFile) {
    WeightMap weightmap("/root/autodl-tmp/llama2_7b_chat.bin");
}