## 介绍

练习项目，基于[fastllm](https://github.com/ztxz16/fastllm)进行了二次开发，目前支持以下特性：
- 支持llama2-7B
- 支持CPU加速
- 支持INT8量化
- 支持BPE、top_p采样
- 无第三方依赖
- 支持Linux
- 代码量~3k行，代码结构简单


|      |   硬件   |   推理速度(tokens/s)   |
| ---- | ---- | ---- |
|  FP16   |   4090   |      |
|  FP16   |    Tesla T4  |      |
|  FP16   |    Intel Xeon  |  1.3    |
|  INT8    |  4090    |      |
|  INT8    |   Tesla T4   |      |
|  INT8    |   Intel Xeon   |      |


## 快速开始
1. 导出模型

```bash
git clone https://github.com/bjmsong/xLLM.git
cd xLLM
conda create --name xllm
conda activate xllm
pip install -r scripts/requirements.txt
huggingface-cli login
python scripts/export_weight.py /pathto/llama2_7b_chat.bin
python scripts/export_tokenizer.py /pathto/tokenizer.bin
```

2.  编译
```bash
mkdir build && cd build
cmake ..
make -j4
```

3. 运行
```bash
./main --weight /pathto/llama2_7b_chat.bin --token /pathto/tokenizer.bin --threads 32
```


5. 量化


## 单元测试

```bash
cmake .. -DDEVELOPMENT=ON
make -j4

./tests/unittest --gtest_filter=test_operator.linear
```