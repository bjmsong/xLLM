#include <gtest/gtest.h>
#include "param.h"

using namespace xllm;

TEST(test_tokenizer, test_tokenizer1) {
    Tokenizer tokenizer("/root/autodl-tmp/tokenizer.bin");
    std::vector<int> tokens;
    tokenizer.Encode("hello world!", tokens);
    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0], );
    ASSERT_EQ(tokens[1], );
    ASSERT_EQ(tokens[2], );
}