from sentence_transformers import SentenceTransformer, util


pretrained_model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
model = SentenceTransformer(pretrained_model_name) # 会自动下载模型

query_embedding = model.encode('北京')
passage_embedding = model.encode(['厨房','故宫', '北京'])

print(query_embedding.shape)
print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))
