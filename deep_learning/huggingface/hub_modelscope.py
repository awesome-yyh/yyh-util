import os
from modelscope import snapshot_download

# https://www.modelscope.cn/

model_name = "qwen/Qwen-7B-Chat"

model_dir = snapshot_download(model_name, revision="master")
print("Model files are saved at: ", model_dir)
