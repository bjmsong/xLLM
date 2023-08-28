#include "model.h"
#include "xllm.h"

struct RunConfig {
    std::string weightPath = "/root/autodl-tmp/llama2_7b_chat.bin";  // 模型文件路径
    std::string tokenPath = "/root/autodl-tmp/tokenizer.bin";
    int threads = 4; // 使用的线程数
    bool lowMemMode = false; // 是否使用低内存模式
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
    std::cout << "<--top_p> <args>:             采样参数top_p" << std::endl;
    std::cout << "<--top_k> <args>:             采样参数top_k" << std::endl;
    std::cout << "<--temperature> <args>:       采样参数温度，越高结果越不固定" << std::endl;
    std::cout << "<--repeat_penalty> <args>:    采样参数重复惩罚" << std::endl;
}

void ParseArgs(int argc, char **argv, RunConfig &config, xllm::GenerationConfig &generationConfig) {
    std::vector <std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        } else if (sargv[i] == "--weight") {
            config.weightPath = sargv[++i];
        } else if (sargv[i] == "--token") {
            config.tokenPath = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--low") {
            config.lowMemMode = true;
        } else if (sargv[i] == "-m" || sargv[i] == "--model") {
            i++;
        } else if (sargv[i] == "--top_p") {
            generationConfig.top_p = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--top_k") {
            generationConfig.top_k = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--temperature") {
            generationConfig.temperature = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--repeat_penalty") {
            generationConfig.repeat_penalty = atof(sargv[++i].c_str());
        } else {
            Usage();
            exit(-1);
        }
    }
}

int main(int argc, char **argv) {
    int round = 0;
    std::string history = "";

    RunConfig config;
    xllm::GenerationConfig generationConfig;
    ParseArgs(argc, argv, config, generationConfig);

    xllm::PrintInstructionInfo();
    xllm::SetThreads(config.threads);
    std::unique_ptr<xllm::LlamaModel> model = std::make_unique<xllm::LlamaModel>(config.weightPath, config.tokenPath);

    printf("输入内容对话, reset清空历史记录, stop退出程序.\n");
    while (true) {
        printf("用户: ");
        std::string input;
        std::getline(std::cin, input);
        if (input == "reset") {
            history = "";
            round = 0;
            continue;
        }
        if (input == "stop") {
            break;
        }
        std::string ret = model->Response(model->MakeInput(history, round, input), [](int index, const char* content) {
            if (index == 0) {
                printf("%s:%s", modelType.c_str(), content);
                fflush(stdout);
            }
            if (index > 0) {
                printf("%s", content);
                fflush(stdout);
            }
            if (index == -1) {
                printf("\n");
            }
        }, generationConfig);
        history = model->MakeHistory(history, round, input, ret);
        round++;
    }

    return 0;
}