# 抽取式 (extractive) 问答：从上下文中截取片段作为回答，类似于我们前面介绍的序列标注任务；
# 生成式 (generative) 问答：生成一个文本片段作为回答，类似于我们前面介绍的翻译和摘要任务。
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


model_name = "/data/app/base_model/uer-roberta-base-chinese-extractive-qa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

question = '世界杯每几年举办一次'
context = '国际足联世界杯以四年为周期，每四年举办一次'

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]

start_index = torch.argmax(start_probabilities)
end_index = torch.argmax(end_probabilities)

# 对其训练时，目标位置label也要同样处理
# 从input_ids中解码（推荐）
print(tokenizer.decode(inputs['input_ids'][0][start_index:end_index]))

# 从原输入文本中根据索引取出
print((question + context)[start_index - 2: end_index - 2])  # 去掉cls和seq
