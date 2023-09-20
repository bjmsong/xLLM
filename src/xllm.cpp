#include "xllm.h"
#include "executor.h"

namespace xllm {
    static int threads = 4;
    static ThreadPool *xllmThreadPool = new ThreadPool(threads);
    Executor defaultExecutor;
    Executor *curExecutor = &defaultExecutor;

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

    void Embedding(const Data &input, Data &weight, Data &output) {
        curExecutor->Run("Embedding", {
                {"input", (Data*)&input}, {"weight", &weight}, {"output", &output}
        }, {}, {});
    }

    void RMSNorm(const Data &input, const Data &weight, Data &output, float eps) {
        curExecutor->Run("RMSNorm", {
                {"input", (Data*)&input}, {"weight", (Data*)&weight}, {"output", &output}
        }, {{"eps", eps}}, {});
    }
    
    void Linear(Data &input, Data &weight, Data &output) {
        curExecutor->Run("Linear", {
                {"input", &input}, {"weight", &weight}, {"output", &output}
        }, {}, {});
    }

    void LlamaRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim) {
        curExecutor->Run("LlamaRotatePosition2D", {
                {"input", &input}, {"positionIds", (Data*)&positionIds}, {"sin", &sinData}, {"cos", &cosData}
        }, {}, {{"rotaryDim", rotaryDim}});
    }

    void PermuteSelf(const Data &input, const std::vector<int> &axis) {
        Data axisData = Data(DataType::INT32PARAM, {(int)axis.size()});
        axisData.Allocate();
        for (int i = 0; i < axisData.Count(0); i++) {
            ((int32_t*)axisData.cpuData)[i] = axis[i];
        }
        curExecutor->Run("PermuteSelf", {
                {"input", (Data*)&input}, {"axis", &axisData}
        }, {}, {});
    }

    void CatDirect(Data &input0, const Data &input1, int axis) {
        curExecutor->Run("CatDirect", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}
        }, {}, {{"axis", axis}});
    }

    void MatMulTransB(const Data &input0, const Data &input1, Data &output, float alpha) {
        curExecutor->Run("MatMulTransB", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}
        }, {{"alpha", alpha}}, {});
    }

    void AttentionMask(Data &input, const Data &mask, float maskValue) {
        curExecutor->Run("AttentionMask", {
                {"input", &input}, {"mask", (Data*)&mask}
        }, {{"maskValue", maskValue}}, {});
    }

    void SoftMax(const Data &input, Data &output, int axis) {
        curExecutor->Run("SoftMax", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {{"axis", axis}});
    }

    void MatMul(const Data &input0, const Data &input1, Data &output, float alpha) {
        curExecutor->Run("MatMul", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}
        }, {{"alpha", alpha}}, {});
    }

    void AddTo(Data &input0, const Data &input1, float alpha) {
        curExecutor->Run("AddTo", {
                {"input0", &input0}, {"input1", (Data*)&input1}
        }, {{"alpha", alpha}}, {});
    }

    void Silu(const Data &input, Data &output) {
        curExecutor->Run("Silu", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {});
    }

    void MulTo(Data &input0, const Data &input1) {
        curExecutor->Run("MulTo", {
                {"input0", &input0}, {"input1", (Data*)&input1}
        }, {}, {});
    }
}