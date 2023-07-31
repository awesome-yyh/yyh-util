# 大部分 Prompt 方法都是通过模板将问题转换为 MLM 任务的形式来解决
# MLM 任务与序列标注任务很相似，也是对 token 进行分类，并且类别是整个词表，不同之处在于 MLM 任务只对文中特殊的 [MASK] token 进行标注
# 因此 MLM 任务的标签同样是一个序列，但是只有 [MASK] token 的位置为对应词语的索引，其他位置都应该设为 -100，以便在使用交叉熵计算损失时忽略它们。
import torch
from transformers import BertTokenizer, AutoModelForMaskedLM


MODEL_PATH = "/data/app/base_model/chinese-roberta-wwm-ext"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)

text = "这个宾馆比较陈旧了，特价的房间也很一般。这种体验很[MASK]。"

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
token_logits = outputs.logits
print(token_logits.shape)  # torch.Size([1, 29, 21128]), 而inputs.input_ids.shape=torch.Size([1, 29]), 即这个logits是每个input token的在词汇表中的映射

# 找到 [MASK] token的索引
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
print(mask_token_index)

# 找到[mask]位置的logits
mask_token_logits = token_logits[0, mask_token_index, :]

# 找到 [MASK]位置的 最可能的n个候选
top_5_tokens_id = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

# 将token id 解码
for token in top_5_tokens_id:
    print(tokenizer.decode([token]))

# 将token id 解码并替换文本[mask]
for token in top_5_tokens_id:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
