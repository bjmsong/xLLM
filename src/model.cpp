#include "model.h"

namespace xllm {

    LlamaModel::LlamaModel(const std::string &weightPath, const std::string &tokenPath): 
        weight(weightPath), tokenizer(tokenPath) {      
        
        pre_prompt = "";
        history_sep = "</s>";

        block_cnt = 32;
        rotary_dim = 128;

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

        std::vector <float> vmask = std::vector <float> (seqLen * seqLen, 0);
        std::vector <float> vpids = std::vector <float> (seqLen, 0);
        for (int i = 0; i < seqLen; i++) {
            vpids[i] = i;
            for (int j = i + 1; j < seqLen; j++) {
                vmask[i * seqLen + j] = 1;
            }
        }

        Data attentionMask = Data(DataType::FLOAT32, {seqLen, seqLen}, vmask);
        Data positionIds = Data(DataType::FLOAT32, {1, seqLen}, vpids);

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
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
            if (ret == eos_token_id) {
                break;
            }

            results.push_back(ret);
            std::string curString = tokenizer.Decode(Data(DataType::FLOAT32, {(int)results.size()}, results)).c_str();
            retString += curString;
            if (retCb)
#ifdef PY_API
			{
				if(generationConfig.enable_hash_id){
					std::stringstream ss;
					ss << retString << "hash_id:"<<hash_id;
					retCb(index, pybind11::bytes(ss.str()));
				}else{
					retCb(index, pybind11::bytes(retString));
				}
			}
#else
                retCb(index, curString.c_str());
#endif
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
#ifdef PY_API
		{
			if(generationConfig.enable_hash_id){
				std::stringstream ss;
				ss << retString << "hash_id:"<<hash_id;
				retCb(-1, pybind11::bytes(ss.str()));
			}else{
				retCb(-1, pybind11::bytes(retString));
			}
		}
#else
            retCb(-1, retString.c_str());
#endif

        return retString;
    }
}