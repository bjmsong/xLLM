#pragma once

#include <memory>
#include <queue>
#include <thread>
#include <mutex>

#include "utils.h"
#include "data.h"
#include "param.h"
#include "xllm.h"
#include "operator.h"

namespace xllm {

    struct LastTokensUnit {
        int tot = 0;
        std::multiset <int> tokenSet;
        std::queue <int> tokenQueue;

        LastTokensUnit () {}

        LastTokensUnit (int tot) {
            Init(tot);
        }

        void Init(int tot) {
            this->tot = tot;
            tokenSet.clear();
            while (tokenQueue.size() > 0) {
                tokenQueue.pop();
            }
        }

        void Push(int id) {
            if (tokenQueue.size() == tot) {
                tokenSet.erase(tokenSet.find(tokenQueue.front()));
                tokenQueue.pop();
            }
            tokenQueue.push(id);
            tokenSet.insert(id);
        }
    };


    struct LastTokensManager {
        std::vector <LastTokensUnit> units;

        LastTokensManager () {}

        LastTokensManager (int batch, int lastN) {
            units.resize(batch);
            for (int i = 0; i < batch; i++) {
                units[i].Init(lastN);
            }
        }
    };

    class LlamaModel {
    public:
        Tokenizer tokenizer;
        WeightMap weight;

        std::string pre_prompt= ""; // 系统设定
        const std::string B_INST{"[INST] "}, E_INST{" [/INST]"}, EOS{""};

        ModelArgs params;

        std::vector<std::vector<float> > sin, cos;
        Data sinData, cosData;

        ResponseContextDict responseContextDict;

        std::thread *mainLoop = nullptr;
        std::mutex mainLoopLocker, dictLocker;

        LlamaModel (const std::string &weightPath, const std::string &tokenPath);
        
        void InitParams(); // 初始化参数信息

        // 推理
        int Forward(
                const Data &inputIds, const Data &attentionMask, const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig,
                const LastTokensManager &lastTokens, std::vector <float> *logits = nullptr);

        std::string Response(const std::vector<float>& input,
                                     RuntimeResult retCb,
                                     const GenerationConfig &generationConfig = GenerationConfig()); // 根据给出的内容回复

        void WarmUp(); // 预热

        std::vector<float> MakeInput(std::vector<float> &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        void MakeHistory(std::vector<float> &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history
    };
}
