from transformers import BertTokenizerFast, BertConfig, BertModel
# 快速分词器除了能进行编码和解码之外，还能够追踪原文到 token 之间的映射，这对于处理序列标注、自动问答等任务非常重要


MODEL_PATH = "bert-base-chinese"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
example = "今天也是充满希望的一天"
encoding = tokenizer(example)
print(encoding.tokens())
print(encoding.word_ids())
