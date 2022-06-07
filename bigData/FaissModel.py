# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import faiss
from AlsEmbedding import getAlsEmbedding
import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py.Logger import Logger
from sklearn import preprocessing

def getFaissModel(filename):
# def getFaissModel(filename, ids, embedding):
  log = Logger("log/als.log")

  ids = pd.read_csv(filename, usecols=[0], squeeze=True, header=None).values.tolist()
  embedding = pd.read_csv(filename, usecols=[2], squeeze=True, header=None).values.tolist()
  embedding = list(map(lambda x:eval(x), embedding))

  ids_np = np.array(ids, dtype = np.int64) # ids需要转成int64类型
  embedding_np = np.array(embedding, dtype = np.float32) # embedding需要转成float32类型，待搜索的embedding也是
  embedding_np_l2 = preprocessing.normalize(embedding_np, norm = 'l2') # l2 正则化

  log.logger.info(f"{filename} faiss模型共有 {ids_np.shape[0]} 条训练数据")

  # 无论使用那个索引，必须使用内积，L2正则化后内积代表余弦相似度，
  dim, measure = embedding_np_l2.shape[1], faiss.METRIC_INNER_PRODUCT # METRIC_L2, METRIC_INNER_PRODUCT

  # # 使用 暴力索引, IndexFlatL2是欧式距离，IndexFlatIP是内积（推荐），最准确，速度慢，可用, 但search耗时75ms
  # index = faiss.IndexFlatIP(dim)
  # indexModel = faiss.IndexIDMap(index) # 将index的id映射到index2的id,会维持一个映射表

  # 使用 IndexIVFFlat 倒排暴力索引, 减小搜索范围，提升速度，可用, search耗时21ms
  # nlist = 100 #聚类中心
  # quantizer  = faiss.IndexFlatIP(dim) # 定义量化器  
  # indexModel = faiss.IndexIVFFlat(quantizer, dim, nlist, measure)
  # indexModel.nprobe = 10

  # # 使用 IndexIVFPQ 索引 将一个向量的维度切成x段，每段分别进行k-means, 可用  
  # nlist = 1000 # 聚类的数目
  # m = 8 # 指定向量要被切分成多少段，所以m一定要能整除向量的维度
  # bits = 8 # 每个子向量用多少个bits表示
  # quantizer = faiss.IndexFlatIP(dim) # 定义量化器  
  # indexModel = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)
  # indexModel.nprobe = 1000 # #查找聚类中心的个数，默认为1个，若nprobe=nlist则等同于精确查找

  # hnsw，基于图检索的改进方法，非常好，还可以分批add
  indexModel = faiss.IndexHNSWFlat(dim, 64, measure) # 第二个参数为构建图时每个点最多连接多少个节点，x越大，构图越复杂，查询越精确，当然构建index时间也就越慢，x取4~64中的任何一个整数
  indexModel.hnsw.efSearch = 64 # efSearch越大越准确，但是速度越慢（不支持GPU）
  indexModel = faiss.IndexIDMap(indexModel)

  log.logger.info(f"is_trained: {indexModel.is_trained}")
  if not indexModel.is_trained: # bool类型，用于指示index是否已被训练
    indexModel.train(embedding_np_l2) # 倒排索引需要训练k-means
  # indexModel.add(embedding) # add_with_ids已经包含add
  indexModel.add_with_ids(embedding_np_l2, ids_np)

  faiss.write_index(indexModel, f"{filename}.index") # 保存模型


if __name__ == "__main__":
  query_filename = "./data/query_embedding.csv"
  doc_filename = "./data/doc_embedding.csv"
  
  # print("开始构建Als Embedding……")
  # query_embedding_map, doc_embedding_map, id_query_map, id_doc_map, id_quEmbedding_map, id_docEmbedding_map = getAlsEmbedding(
  #     "20220313", "20220313", query_filename, doc_filename)
  
  print("开始构建query-faiss索引……")
  getFaissModel(query_filename)
  # ids = list(id_quEmbedding_map.keys())
  # embedding = list(id_quEmbedding_map.values())
  # getFaissModel(query_filename, ids, embedding)
  
  print("开始构建doc-faiss索引……")
  getFaissModel(doc_filename)
  # ids = list(id_docEmbedding_map.keys())
  # embedding = list(id_docEmbedding_map.values())
  # getFaissModel(doc_filename, ids, embedding)
