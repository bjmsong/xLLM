#pragma once

#include <memory>
#include "cmath"
#include <iostream>

#include "utils.h"
#include "data.h"
#include "param.h"
#include "xllm.h"

namespace xllm {
    class LlamaModel {
    public:
        std::string pre_prompt; // 最初对话的提示语
        std::string user_role, bot_role, history_sep; // 用于生成每一轮的prompt
        const std::string B_INST{"[INST] "}, E_INST{" [/INST]"};

        int bos_token_id;
        int eos_token_id;
        int embed_dim = 4096;
        int num_attention_heads = 32;
        int head_dim = embed_dim / num_attention_heads;
        const int max_positions = 32768;
        int rotary_dim = 64;
        const float scale_attn = sqrt(head_dim);
        int block_cnt = 28;

        std::vector<std::vector<float> > sin, cos;
        Data sinData, cosData;

        Tokenizer tokenizer;
        WeightMap weight;

        ResponseContextDict responseContextDict;

        std::thread *mainLoop = nullptr;
        std::mutex mainLoopLocker, dictLocker;

        LlamaModel (const std::string &weightPath, const std::string &tokenPath);
        
        void InitParams(); // 初始化参数信息

        // 推理
        int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager());

        std::string Response(const std::string& input,
                                     RuntimeResult retCb,
                                     const GenerationConfig &generationConfig = GenerationConfig()); // 根据给出的内容回复

        void WarmUp(); // 预热

        std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history
    };
}
