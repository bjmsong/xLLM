#include "xllm.h"

namespace xllm {
    static int threads = 4;
    static ThreadPool *xllmThreadPool = new ThreadPool(threads);

    void PrintInstructionInfo() {
        std::string avx = "OFF", avx2 = "OFF";
#ifdef __AVX__
        avx = "ON";
#endif
#ifdef __AVX2__
        avx2 = "ON";
#endif
        printf("AVX: %s\n", avx.c_str());
        printf("AVX2: %s\n", avx2.c_str());
    }

    void SetThreads(int t) {
        threads = t;
        if (xllmThreadPool) delete xllmThreadPool;
        xllmThreadPool = new ThreadPool(t);
    }

    int GetThreads() {
        return threads;
    }

    ThreadPool *GetPool() {
        return xllmThreadPool;
    }

}