import os
from huggingface_hub import snapshot_download

# snapshot_download的内部使用了hf_hub_download()%
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 自带下载进度条，下载到指定目录
downloaded = []

snapshot_download(repo_id=model_name, ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.safetensors", "*onnx*"] + downloaded, local_dir=os.path.join("~/mypython", model_name.replace('/', '-')), local_dir_use_symlinks=False)
# (推荐) 只下载Pytorch版本的模型, 并指定下载路径, 如果文件已经缓存则直接从缓存文件复制, 要使用local_dir参数需要版本huggingface_hub==0.13.4或以上

# # 只下载指定的文件
# snapshot_download(repo_id=model_name, allow_patterns='vocab.txt', local_dir=os.path.join("/data/app/base_model", model_name.replace('/', '-')), local_dir_use_symlinks=False)  # allow_patterns也可以像ignore_patterns使用正则表达式
