# 大部分 Prompt 方法都是通过模板将问题转换为 MLM 任务的形式来解决
# MLM 任务与序列标注任务很相似，也是对 token 进行分类，并且类别是整个词表，不同之处在于 MLM 任务只对文中特殊的 [MASK] token 进行标注
# 因此 MLM 任务的标签同样是一个序列，但是只有 [MASK] token 的位置为对应词语的索引，其他位置都应该设为 -100，以便在使用交叉熵计算损失时忽略它们。
import os
import random
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import fastseq


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


seed_everything()
MODEL_PATH = "bert-base-chinese"
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

t1 = time.time()
# text = "这个宾馆比较陈旧了，特价的房间也很一般。这种体验很[MASK]。"
# text = "这个宾馆比较陈旧了，体验很差，这个宾馆比较陈旧了，特价的房间也很一般。这种体验很[MASK]。"
text = "这家宾馆很糟，这个宾馆比较陈旧了，体验很差，这个宾馆比较陈旧了，特价的房间也很一般。这种体验很[MASK]。"

inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
print(token_logits.shape)
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
print(mask_token_index)
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

print(time.time() - t1)
