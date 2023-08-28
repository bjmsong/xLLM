#pragma once
#include <string>
#include <set>
#include <unordered_map>
#include <map>

#include "data.h"

namespace xllm {

struct Tokenizer {

    unsigned int vocab_size = 1;
    int bos_id, eos_id;

    std::unordered_map<std::string, int> token_id;
    std::vector<std::string> id_token;
    std::vector<float> id_score;

    Tokenizer (const std::string path);

    Data Encode(const std::string &s, bool bos = false, bool eos = false);

    std::string Decode(const Data& data);
};

struct WeightMap {

    std::unordered_map <std::string, std::string> params;

    std::map <std::string, Data> weight;

    WeightMap(const std::string &fileName);

    void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型, bit = 0代表直接存

    Data &operator [] (const std::string &key);
};

}