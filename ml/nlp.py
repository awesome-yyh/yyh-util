# from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2') # 会自动下载模型

# query_embedding = model.encode('北京')
# passage_embedding = model.encode(['厨房','故宫', '北京'])

# print(query_embedding.shape)
# print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))



from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

# prepare input
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

# forward pass
output = model(**encoded_input).logits
print(output)
print(output.shape)