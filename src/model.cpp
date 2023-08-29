#include "model.h"

namespace xllm {

    LlamaModel::LlamaModel(const std::string &weightPath, const std::string &tokenPath): 
        weight(weightPath), tokenizer(tokenPath) {      
        
        sin.resize(max_positions);
        cos.resize(max_positions);
        std::vector <float> invFreq;
        for (int i = 0; i < rotary_dim; i += 2) {
            invFreq.push_back(1.0 / pow(10000, (float)i / rotary_dim));
        }
        for (int i = 0; i < max_positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < invFreq.size(); j++) {
                sin[i][j] = ::sin((float)i * invFreq[j]);
                cos[i][j] = ::cos((float)i * invFreq[j]);
            }
        }
        std::vector <float> fsin, fcos;
        for (int i = 0; i < sin.size(); i++) {
            for (int j = 0; j < sin[0].size(); j++) {
                fsin.push_back(sin[i][j]);
                fcos.push_back(cos[i][j]);
            }
        }
        sinData.CopyFrom(Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, fsin));
        cosData.CopyFrom(Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, fcos));
        
        WarmUp();
    }

    std::vector<float> LlamaModel::MakeInput(std::vector<float> &history, int round, const std::string &input) {
        std::string input_trim = trim(input);
        std::string prompt = (round == 0 ? pre_prompt : "") + B_INST + input_trim + E_INST;
        std::vector<float> inputIds = tokenizer.Encode(input, true);
        history.insert(history.end(), inputIds.begin(), inputIds.end());
        return history;
    }

    void LlamaModel::MakeHistory(std::vector<float> &history, int round, const std::string &input, const std::string &output) {
        std::string input_trim = trim(input);
        std::string output_trim = trim(output);
        std::string last =  B_INST + input_trim + E_INST + output_trim;
        std::vector<float> lastchat = tokenizer.Encode(last, true ,true);
        history.insert(history.end(), lastchat.begin(), lastchat.end());
    }

    std::string LlamaModel::Response(const std::vector<float>& input, RuntimeResult retCb,
                                     const GenerationConfig &generationConfig) {

        auto st = std::chrono::system_clock::now();

        std::vector <float> ids;
        int seqLen = input.size();
        Data inputIds(DataType::FLOAT32, {1, seqLen}, input);

        std::vector <float> vmask = std::vector <float> (seqLen * seqLen, 0);   // mask matrix
        std::vector <float> vpids = std::vector <float> (seqLen, 0);
        for (int i = 0; i < seqLen; i++) {
            vpids[i] = i;
            for (int j = i + 1; j < seqLen; j++) {
                vmask[i * seqLen + j] = 1;
            }
        }
        Data attentionMask = Data(DataType::FLOAT32, {seqLen, seqLen}, vmask);
        Data positionIds = Data(DataType::FLOAT32, {1, seqLen}, vpids);

        std::vector <std::pair <Data, Data> > pastKeyValues;  // KV cache
        for (int i = 0; i < params.block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32), Data(DataType::FLOAT32)));
        }

        std::string retString = "";
        int len = seqLen;
        std::vector <float> results;
        int index = 0;

        LastTokensManager tokens (1, generationConfig.last_n);
        while (true) {
            auto st = std::chrono::system_clock::now();

            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
            tokens.units[0].Push(ret);
            if (ret == tokenizer.eos_id) {
                break;
            }

            results.push_back(ret);
            std::string curString = tokenizer.Decode(Data(DataType::FLOAT32, {(int)results.size()}, results)).c_str();
            retString += curString;
            if (retCb)
                retCb(index, curString.c_str());
            index++;

            if (index == generationConfig.output_token_limit) {
                break;
            }
            results.clear();

            attentionMask.ToDevice(DataDevice::CPU);
            positionIds.ToDevice(DataDevice::CPU);
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)ret}));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)len}));
            //if (do_sample) {
            //    tokenPenaltyManager.InsertToken(ret);
            //}
            len++;
            if (index == generationConfig.output_token_limit) {
                break;
            }

            printf("spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        }
        if (retCb)
            retCb(-1, retString.c_str());

        return retString;
    }

    int LlamaModel::Forward(const Data &inputIds, const Data &attentionMask, const Data &positionIds, 
                            std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    1e-6, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int per = qkv.dims.back() / 3;
                Split(qkv, -1, 0, per, q);
                Split(qkv, -1, per, per * 2, k);
                Split(qkv, -1, per * 2, per * 3, v);
            } else {
                Linear(attenInput, weight[qWeightName], Data(), q);
                Linear(attenInput, weight[kWeightName], Data(), k);
                Linear(attenInput, weight[vWeightName], Data(), v);
            }

            std::vector <int> qkvSize = {bsz, seqlen, num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            if (alibiData.dims.size() == 0) {
                fastllm::LlamaRotatePosition2D(q, positionIds, sinData, cosData, rotary_dim);
                fastllm::LlamaRotatePosition2D(k, positionIds, sinData, cosData, rotary_dim);
            }

            qkvSize = {bsz * seqlen, num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            PermuteSelf(q, {1, 0, 2});
            PermuteSelf(k, {1, 0, 2});
            PermuteSelf(v, {1, 0, 2});

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            int unitLen = 64;
            while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || k.dims[1] > pastKey.expansionDims[1]))
                   || (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector <int> {k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }
            while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                   || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }
            CatDirect(pastKey, k, 1);
            CatDirect(pastValue, v, 1);

            // 1.2 Attention
            // 1.2.0 q * k^T
            MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim));
            attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
            if (attentionMask.dims.size() != 0) {
                AttentionMask(attenWeights, attentionMask, -10000);
            }

            Softmax(attenWeights, attenWeights, -1);
            MatMul(attenWeights, pastValue, attenOutput);

            attenOutput.Reshape({attenOutput.dims[1], attenOutput.dims[2], attenOutput.dims[3]});
            PermuteSelf(attenOutput, {1, 0, 2});
            attenOutput.Reshape({bsz, seqlen, -1});

            Linear(attenOutput, weight[oWeightName], Data(), attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], 1e-6, attenInput);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);
            AddTo(hiddenStates, w2);
        }

        RMSNorm(hiddenStates, weight["model.norm.weight"], 1e-6, hiddenStates);
        Data logits;
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        logits.ToDevice(DataDevice::CPU);

        int lastRet = -1;
        if (generationConfig.output_logits && retLogits != nullptr) {
            int size = logits.dims.back();
            logits.ToDevice(DataDevice::CPU);
            retLogits->resize(size);
            memcpy((float*)retLogits->data(), ((float*)logits.cpuData) + (logits.dims[1] - 1) * size, size * logits.unitSize);
        }
        if (generationConfig.IsSimpleGreedy()) {
            std::pair <float, int> ret = std::make_pair(-1e9, -1);
            int base = logits.dims[1] - 1;
            for (int i = 0; i < logits.dims.back(); i++) {
                ret = max(ret, std::make_pair(((float*)logits.cpuData)[base * logits.dims.back() + i], i));
            }
            lastRet = ret.second;
        } else if (!lastTokens.units.empty()) {
            lastRet = LLMSampling(logits, logits.dims[1] - 1, generationConfig, lastTokens.units[0]);
        }

        return lastRet;
    }
}