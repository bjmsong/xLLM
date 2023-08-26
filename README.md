1. 导出模型文件

```bash
pip install -r requirements.txt
huggingface-cli login
python scripts/export.py /root/autodl-tmp/llama2_7b_chat.bin
```

2.  编译
mkdir build & cd build
cmake ..
make -j4

3. 运行