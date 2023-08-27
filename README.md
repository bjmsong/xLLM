1. 导出模型文件、tokenizer

```bash
git clone https://github.com/bjmsong/xLLM.git
cd xLLM
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
./main --path /pathto/llama2_7b_chat.bin --token /pathto/tokenizer.bin
```