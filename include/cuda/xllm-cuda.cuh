#pragma once
#include "data.h"

void *xllmCudaMalloc(size_t size);
void xllmCudaFree(void *ret);
void xllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size);
void xllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size);
void *xllmCudaPrepareInput(const fastllm::Data &input);
void xllmCudaFinishInput(const fastllm::Data &input, void *data);
void xllmCudaSetDevice(int gpu_id);

bool xllmCudaSilu(const Data &input, Data &output);
bool xllmCudaSoftmax(const Data &input, Data &output, int axis);
bool xllmCudaAddTo(Data &input0, const Data &input1, float alpha);
bool xllmCudaMulTo(Data &input0, const Data &input1, float alpha);
bool xllmCudaAttentionMask(Data &input, const Data &mask, float maskValue);
bool xllmCudaRMSNorm(const Data &input, Data &weight, Data &output, float eps);
bool xllmCudaLayerNorm(const Data &input, Data &gamma, Data &beta, Data &output, int axis);
bool xllmCudaPermute(Data &input, const std::vector<int> &axis);
bool xllmCudaMatMulFloatInt8(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k);
bool xllmCudaMatMulFloat16(const Data &input, Data &weight, const Data &bias, Data &output, int n, int m, int k);
