#include "param.h"
#include "utils.h"
#include "file.h"

namespace xllm{
    Tokenizer::Tokenizer(const std::string path){
        FileReader reader(path);
        vocab_size = reader.ReadInt();
        max_token_length = reader.ReadInt();

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
    
    Data Tokenizer::Encode(const std::string &s, std::vector<int>& tokens) {

        // first encode every individual byte in the input string
        for (char c: s) 
            tokens.push_back(token_id[std::string{c}]);

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        int n_tokens = s.length(); // the number of tokens
        while (1) {
            float best_score = -1e10;
            int best_id = -1;
            int best_idx = -1;

            for (int i=0; i < n_tokens-1; i++) {
                int id = token_id[id_token[tokens[i]] + id_token[tokens[i+1]]];
                if (id_score[id] > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = id_score[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (int i = best_idx+1; i < n_tokens-1; i++) {
                tokens[i] = tokens[i+1];
            }
            n_tokens--; // token length decreased
        }
    }
    
}