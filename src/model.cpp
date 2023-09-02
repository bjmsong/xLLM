#include "model.h"
#include "data.h"

namespace xllm {

    LlamaModel::LlamaModel(const std::string &weightPath, const std::string &tokenPath): 
        weight(weightPath), tokenizer(tokenPath) {      
        
        sin.resize(params.max_positions);
        cos.resize(params.max_positions);
        std::vector <float> invFreq;
        for (int i = 0; i < params.rotary_dim; i += 2) {
            invFreq.push_back(1.0 / pow(10000, (float)i / params.rotary_dim));
        }
        for (int i = 0; i < params.max_positions; i++) {
            sin[i].resize(params.rotary_dim);
            cos[i].resize(params.rotary_dim);
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
        sinData.CopyFrom(Data(DataType::FLOAT32, {(int)sin.size(), (int)sin[0].size()}, fsin));
        cosData.CopyFrom(Data(DataType::FLOAT32, {(int)cos.size(), (int)cos[0].size()}, fcos));
        
        // WarmUp();
    }

    std::vector<float> LlamaModel::MakeInput(std::vector<float> &history, int round, const std::string &input) {
        std::string input_trim = trim(input);
        std::string prompt = (round == 0 ? pre_prompt : "") + B_INST + input_trim + E_INST;
        std::vector<float> inputIds = tokenizer.Encode(prompt, true);
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

    void LlamaModel::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < params.block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        printf("finish.\n");
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
                vmask[i * seqLen + j] = 1;   // mask标记为1
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
            // auto st = std::chrono::system_clock::now();

            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);
            tokens.units[0].Push(ret);
            if (ret == tokenizer.eos_id) {
                break;
            }

            results.push_back(ret);
            std::string curString = tokenizer.Decode(Data(DataType::FLOAT32, {(int)results.size()}, results));
            retString += curString;
            if (retCb)
                retCb(index, curString.c_str());
            index++;

            if (index == generationConfig.output_token_limit) {
                break;
            }
            results.clear();
            
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)ret}));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)len}));
            len++;
            if (index == generationConfig.output_token_limit) {
                break;
            }

            // printf("spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        }
        if (retCb)
            retCb(-1, retString.c_str());

        return retString;
    }

    int LlamaModel::Forward(const Data &inputIds, const Data &attentionMask, const Data &positionIds, 
                            std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        
        int bsz = 1, seqlen = inputIds.dims[1];
        Data hiddenStates(DataType::FLOAT32, {bsz, seqlen, params.embed_dim});
        Embedding(inputIds, weight["model.embed_tokens.weight"], hiddenStates);

        Data attenInput(DataType::FLOAT32, hiddenStates.dims);
        Data q(DataType::FLOAT32, hiddenStates.dims), k(DataType::FLOAT32, hiddenStates.dims), v(DataType::FLOAT32, hiddenStates.dims);
        Data attenOutput(DataType::FLOAT32, {bsz, params.num_attention_heads, bsz* seqlen, params.hidden_size/params.num_attention_heads});
        Data attenLastOutput(DataType::FLOAT32, {bsz, seqlen, params.hidden_size});
        Data w1(DataType::FLOAT32, {bsz, seqlen, params.intermediate_size});
        Data w3(DataType::FLOAT32, {bsz, seqlen, params.intermediate_size});
        Data w2(DataType::FLOAT32, {bsz, seqlen, params.hidden_size});
        for (int i = 0; i < params.block_cnt; i++) {
            RMSNorm(hiddenStates, weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    attenInput, 1e-6);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";

            // 1.1 Get q, k, v
            q.Reshape(hiddenStates.dims);
            k.Reshape(hiddenStates.dims);
            v.Reshape(hiddenStates.dims);
            Linear(attenInput, weight[qWeightName], q);
            Linear(attenInput, weight[kWeightName], k);
            Linear(attenInput, weight[vWeightName], v);

            std::vector <int> qkvSize = {bsz, seqlen, params.num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);

            LlamaRotatePosition2D(q, positionIds, sinData, cosData, params.rotary_dim);
            LlamaRotatePosition2D(k, positionIds, sinData, cosData, params.rotary_dim);

            qkvSize = {bsz * seqlen, params.num_attention_heads, -1};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            std::vector<int> axisData = {1, 0, 2};
            PermuteSelf(q, axisData);
            PermuteSelf(k, axisData);
            PermuteSelf(v, axisData);

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            int unitLen = 64;   // 每次扩容的seq_len是unitLen的倍数
            while ((pastKey.dims.size() == 0 && (pastKey.expandDims.size() == 0 || k.dims[1] > pastKey.expandDims[1]))
                   || (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expandDims[1])) {
                std::vector <int> newDims;
                if (pastKey.counts == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector <int> {k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }
            while ((pastValue.dims.size() == 0 && (pastValue.expandDims.size() == 0 || v.dims[1] > pastValue.expandDims[1]))
                   || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expandDims[1])) {
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
            // q: {num_attention_heads, bsz * seqlen, hidden_size/num_attention_heads}
            // pastKey: {num_attention_heads, k_seqlen, hidden_size/num_attention_heads} 不同head之间的内存不连续
            Data attenWeights(DataType::FLOAT32, {q.dims[0], q.dims[1], pastKey.dims[1]});
            MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(params.head_dim));
            attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
            // 只有第一轮需要mask
            if (attentionMask.dims.size() != 0) {
                AttentionMask(attenWeights, attentionMask, -10000);
            }

            SoftMax(attenWeights, attenWeights, -1);
            // attenWeights: {1, num_attention_heads, bsz * seqlen, k_seqlen}
            // pastValue: {num_attention_heads, k_seqlen, hidden_size/num_attention_heads} 不同head之间的内存不连续
            // attenOutput: {1, num_attention_heads, bsz * seqlen, hidden_size/num_attention_heads}
            attenOutput.Reshape({1, params.num_attention_heads, bsz * seqlen, -1});
            MatMul(attenWeights, pastValue, attenOutput);

            attenOutput.Reshape({attenOutput.dims[1], attenOutput.dims[2], attenOutput.dims[3]});
            PermuteSelf(attenOutput, axisData);  // 这里为啥要转置
            // {bsz, seqLen, hidden_size}
            attenOutput.Reshape({bsz, seqlen, -1});

            // weight[oWeightName]: {hidden_size, hidden_size}
            Linear(attenOutput, weight[oWeightName], attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);

            // 2. mlp
            RMSNorm(hiddenStates, weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], attenInput, 1e-6);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"],  w1);
            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"],  w3);
            Silu(w1, w1);
            MulTo(w1, w3);
            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], w2);
            AddTo(hiddenStates, w2);
        }

        RMSNorm(hiddenStates, weight["model.norm.weight"], hiddenStates, 1e-6);
        Data logits(DataType::FLOAT32, {bsz, hiddenStates.dims[1], params.vocab_size});
        Linear(hiddenStates, weight["lm_head.weight"], logits);

        // 采样
        int lastRet = -1;

        if (generationConfig.IsSimpleGreedy()) {
            std::pair <float, int> ret = std::make_pair(-1e9, -1);
            int base = logits.dims[1] - 1;
            for (int i = 0; i < logits.dims.back(); i++) {
                ret = max(ret, std::make_pair(((float*)logits.cpuData)[base * logits.dims.back() + i], i));
            }
            lastRet = ret.second;
        }
        
        // int base = logits.dims[1] - 1;
        
        // return sample_top_p(logits.cpuData[base * logits.dims.back()], generationConfig);
    }

    int LlamaModel::compare_indexed_float(const void* a, const void* b) {
        IndexedFloat* indexed_a = (IndexedFloat*)a;
        IndexedFloat* indexed_b = (IndexedFloat*)b;
        return (indexed_b->value > indexed_a->value) ? 1 : ((indexed_b->value < indexed_a->value) ? -1 : 0);
    }

    int LlamaModel::sample_top_p(float* probs, const GenerationConfig& generationConfig) {
        int n = params.vocab_size;
        IndexedFloat probs_sort[n];
        for (int i = 0; i < n; i++) {
            probs_sort[i].value = probs[i];
            probs_sort[i].index = i;
        }
        qsort(probs_sort, n, sizeof(IndexedFloat), compare_indexed_float);

        float accum = 0.f;
        int p = 0;
        for (; accum<=generationConfig.top_p && p < n; p++) {
            accum += probs_sort[p].value;
        }

        // random float32 in [0,1)
        std::random_device rd; // 用于生成随机种子的设备
        std::mt19937 gen(rd()); // Mersenne Twister 伪随机数生成器
        std::uniform_real_distribution<float> dis(0.0f, 1.0f); // 均匀分布在[0, 1)之间的浮点数

        float r = dis(gen) * accum;
        float cdf = 0.0f;
        for (int i = 0; i < p; i++) {
            cdf += probs_sort[i].value;
            if (r < cdf) {
                return probs_sort[i].index;
            }
        }
        return probs_sort[p-1].index; // in case of rounding errors
    }
}