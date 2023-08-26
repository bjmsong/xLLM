1. 导出模型文件

```bash
git clone https://github.com/bjmsong/xLLM.git
cd xLLM
pip install -r scripts/requirements.txt
huggingface-cli login
python scripts/export.py /pathto/llama2_7b_chat.bin
```

2.  编译
```bash
mkdir build & cd build
cmake ..
make -j4
```

3. 运行