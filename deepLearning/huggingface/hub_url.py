from huggingface_hub import hf_hub_url
from huggingface_hub.utils import filter_repo_objects
from huggingface_hub.hf_api import HfApi

repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
repo_type = "model"  # 如果是数据 dataset

repo_info = HfApi().repo_info(repo_id=repo_id, repo_type=repo_type)  # 有时候会连接Error，多试几次
files = list(filter_repo_objects(items=[f.rfilename for f in repo_info.siblings]))
urls = [hf_hub_url(repo_id, filename=file, repo_type=repo_type) for file in files]
print("\n".join(urls))
