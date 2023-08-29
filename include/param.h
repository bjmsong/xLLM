#pragma once
#include <string>
#include <set>
#include <unordered_map>

#include "data.h"

namespace xllm {

    struct Tokenizer {

        unsigned int vocab_size = 1;
        int bos_id, eos_id;

        std::unordered_map<std::string, int> token_id;
        std::vector<std::string> id_token;
        std::vector<float> id_score;

        Tokenizer (const std::string path);

        std::vector<float> Encode(const std::string &s, bool bos = false, bool eos = false);

        std::string Decode(const Data& data);
    };

    struct WeightMap {

        std::unordered_map <std::string, std::string> params;

        std::map <std::string, Data> weight;

        WeightMap(const std::string &fileName);

        void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型, bit = 0代表直接存

        Data &operator [] (const std::string &key);
    };

    struct ModelArgs{
        int block_cnt = 32;
        int embed_dim = 4096;
        int num_attention_heads = 32;
        int head_dim = embed_dim / num_attention_heads;
        int vocab_size = 32000;
        int rotary_dim = 128;  // 官方代码中没有找到
        const float scale_attn = sqrt(head_dim);
    };
}