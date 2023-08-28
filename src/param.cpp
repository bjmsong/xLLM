#include "param.h"
#include "utils.h"
#include "file.h"

namespace xllm{
    Tokenizer::Tokenizer(const std::string path){
        FileReader reader(path);
        vocab_size = reader.ReadInt();
        bos_id = reader.ReadInt();
        eos_id = reader.ReadInt();

        float score;
        std::string token;
        for (int i = 0; i < vocab_size; i++) {
            score = reader.ReadFloat();
            token = reader.ReadString();
            token_id[token] = i;
            id_score.push_back(score);
            id_token.push_back(token);
        }
    }
    
    Data Tokenizer::Encode(const std::string &s, bool bos, bool eos) {

        std::vector<int> tokens;
        // first encode every individual byte in the input string
        for (char c: s) 
            tokens.push_back(token_id[std::string{c}]);

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        int n_tokens = s.length(); // the number of tokens
        while (1) {
            float best_score = -1e10;
            int best_id = -1;
            int best_idx = -1;

            std::string merString;
            for (int i=0; i < n_tokens-1; i++) {
                merString = id_token[tokens[i]] + id_token[tokens[i+1]];
                if (token_id.find(merString) != token_id.end()){
                    int id = token_id[merString];
                    if (id_score[id] > best_score) {
                        // this merge pair exists in vocab! record its score and position
                        best_score = id_score[id];
                        best_id = id;
                        best_idx = i;
                    }
                }
            }

            if (best_idx == -1)
                break; // we couldn't find any more pairs to merge, so we're done

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (int i = best_idx+1; i < n_tokens-1; i++) {
                tokens[i] = tokens[i+1];
            }
            n_tokens--; // token length decreased
        }

        std::vector<float> tokens_output;
        if(bos)
            tokens_output.push_back(bos_id);
        for(int i=0; i<n_tokens; i++)
            tokens_output.push_back(tokens[i]);
        if(eos)
            tokens_output.push_back(eos_id);
        
        return Data(DataType::FLOAT32, {n_tokens + bos + eos}, tokens_output);
    }
        

    std::string Tokenizer::Decode(const Data& data) {
        std::vector <int> tokens;
        for (int i = 0; i < data.counts; i++) {
            tokens.push_back((int) ((float *) data.cpuData)[i]);
        }

        

        return "";
    }

    WeightMap::WeightMap(const std::string &fileName){
        FileReader buffer(fileName);

        int keyValueLen = buffer.ReadInt();
        for (int i = 0; i < keyValueLen; i++) {
            std::string key = buffer.ReadString();
            std::string value = buffer.ReadString();
            params[key] = value;
        }

        for(auto& pair:params)
            std::cout << "Key: " << pair.first << " Value: " << pair.second << std::endl;
    
        int weightLen = buffer.ReadInt();
        for (int i = 0; i < weightLen; i++) {
            std::string name = buffer.ReadString();
            //printf("%s\n", name.c_str());
            int dimsSize = buffer.ReadInt();
            //printf("size = %d\n", dimsSize);
            std::vector <int> dims;
            for (int j = 0; j < dimsSize; j++) {
                int x = buffer.ReadInt();
                dims.push_back(x);
            }
            DataType dataType = (DataType)buffer.ReadInt();
            weight[name] = Data(dataType, dims);
            weight[name].Allocate();
            if (dataType == DataType::FLOAT32 || dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
                buffer.ReadBytes(weight[name].cpuData, weight[name].bytes);
            }

            printf("Load (%d / %d) \r", (i + 1), weightLen);
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }

    Data &WeightMap::operator[](const std::string &key) {
        return weight[key];
    }
}