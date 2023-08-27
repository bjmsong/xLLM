#pragma once
#include <string>
#include <set>
#include <unordered_map>

#include "data.h"

namespace xllm {

struct Tokenizer {

    int max_token_length = 1;
    unsigned int vocab_size = 1;
    
    std::unordered_map<std::string, int> token_id;
    std::vector<std::string> id_token;
    std::vector<float> id_score;

    Tokenizer (const std::string path);

    void Encode(const std::string &s, std::vector<int>& tokens);
};

// struct WeightMap {
//     int versionId = 2;

//     Tokenizer tokenizer(std::string path);

//     std::map <std::string, std::string> dicts;

//     std::map <std::string, Data> weight;

//     std::set <std::string> embeddingNames;

//     void LoadFromFile(const std::string &fileName); // 从文件读取

//     void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型, bit = 0代表直接存

//     void AddTokenizerWord(const std::string &key, int value); // 增加一个词

//     void AddDict(const std::string &key, const std::string &value); // 插入一个词条

//     void AddWeight(const std::string &key, const std::vector <int> &dims,
//                     DataType dataType, WeightType weightType, DataType oriDataType, uint8_t *oriData); // 插入一个权重

//     Data &operator [] (const std::string &key);
// };

}