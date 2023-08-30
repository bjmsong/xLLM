基于[fastllm](https://github.com/ztxz16/fastllm)进行了二次开发，目前支持以下特性：
- 支持llama2-7B
- 支持Linux
- 支持CPU加速
- 支持INT8量化
- 支持BPE
- 无第三方依赖
- 代码量少于2k行，代码结构简单

欢迎star！

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
./main --weight /pathto/llama2_7b_chat.bin --token /pathto/tokenizer.bin
```


5. 量化

10. 开启单元测试
```bash
cmake .. -DDEVELOPMENT=ON
make -j4

./test/unittest --gtest_filter=test_operator.*
```