# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
import os, sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer
from operator import itemgetter


os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def getFaissModel(filename):
    emb_df = pd.read_csv(filename, header=None)  # id, 句子, embedding
    
    ids_np = np.array(emb_df[0], dtype=np.int64)  # ids需要转成int64类型
    embedding = list(map(lambda x: eval(x), emb_df[2]))
    embedding_np = np.array(embedding, dtype=np.float32)  # embedding需要转成float32类型，待搜索的embedding也是
    embedding_np_l2 = preprocessing.normalize(embedding_np, norm='l2')  # l2 正则化

    print(f"{filename} faiss模型共有 {ids_np.shape[0]} 条训练数据")

    # 无论使用那个索引，必须使用内积，L2正则化后内积代表余弦相似度，
    dim, measure = embedding_np_l2.shape[1], faiss.METRIC_INNER_PRODUCT  # METRIC_L2, METRIC_INNER_PRODUCT

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
    indexModel = faiss.IndexHNSWFlat(dim, 64, measure)  # 第二个参数为构建图时每个点最多连接多少个节点，x越大，构图越复杂，查询越精确，当然构建index时间也就越慢，x取4~64中的任何一个整数
    indexModel.hnsw.efSearch = 64  # efSearch越大越准确，但是速度越慢（不支持GPU）
    indexModel = faiss.IndexIDMap(indexModel)

    print(f"is_trained: {indexModel.is_trained}")
    if not indexModel.is_trained:  # bool类型，用于指示index是否已被训练
        indexModel.train(embedding_np_l2)  # 倒排索引需要训练k-means
    # indexModel.add(embedding) # add_with_ids已经包含add
    indexModel.add_with_ids(embedding_np_l2, ids_np)

    faiss_file = f"{filename}.index"
    faiss.write_index(indexModel, faiss_file)  # 保存模型
    print(f"faiss索引已保存在: {faiss_file}")


if __name__ == "__main__":
    # 根据embedding文件构建索引
    embedding_file = "data/similar_emb.csv"  # id, 句子, embedding
    faiss_file = f"{embedding_file}.index"
    if not Path(faiss_file).exists():
        print("开始构建faiss索引……")
        getFaissModel(embedding_file)
    
    # 读取faiss索引文件，给目标文件匹配相似句子组
    faiss_index_model = faiss.read_index(faiss_file)
    
    emb_df = pd.read_csv(embedding_file, header=None)
    id2sens = {index: row[1] for index, row in emb_df.iterrows()}  # id: 句子

    sentence_model = SentenceTransformer("/data/app/base_model/sentence-transformers_distiluse-base-multilingual-cased-v2")  # 512维
    
    def get_similar_sens(row_sens):
        test_query_emb = sentence_model.encode([row_sens])
        Distance, Index = faiss_index_model.search(test_query_emb, 16)
        sens_tuple = itemgetter(*Index[0])(id2sens)
        # print(Index[0], sens_tuple)
        return sens_tuple
    
    sens_pair_txt = "data/sighan15.txt"  # 序号   原句    正确句  label
    sens_pair_df = pd.read_csv(sens_pair_txt, sep='\t', header=None, usecols=[1, 2])
    
    tqdm.pandas()
    sens_pair_df[3] = sens_pair_df[1].progress_apply(get_similar_sens)  # 给原句匹配相似句
    # print(sens_pair_df.head())
    out_file = sens_pair_txt[:-4] + "_similars.txt"
    sens_pair_df.to_csv(out_file, sep='\t', encoding='utf_8_sig', header=False, index=False)
    
    print("相似句子查找完毕, 在文件: ", out_file)
