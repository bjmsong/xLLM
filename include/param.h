#pragma once
#include <string>
#include <map>
#include <set>
#include "data.h"

namespace xllm {

struct Tokenizer {

    Tokenizer (std::string path);

    ~Tokenizer();

    Data Encode(const std::string &s); // 编码

    std::string Decode(const Data &data); // 解码

};

struct WeightMap {
    int versionId = 2;

    Tokenizer tokenizer(std::string path);

    std::map <std::string, std::string> dicts;

    std::map <std::string, Data> weight;

    std::set <std::string> embeddingNames;

    void LoadFromFile(const std::string &fileName); // 从文件读取

    void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型, bit = 0代表直接存

    void AddTokenizerWord(const std::string &key, int value); // 增加一个词

    void AddDict(const std::string &key, const std::string &value); // 插入一个词条

    void AddWeight(const std::string &key, const std::vector <int> &dims,
                    DataType dataType, WeightType weightType, DataType oriDataType, uint8_t *oriData); // 插入一个权重

    Data &operator [] (const std::string &key);
};

}