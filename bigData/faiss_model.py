import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from sklearn import preprocessing
from operator import itemgetter
sys.path.append(str(Path(__file__).parent.parent.absolute()))


os.environ["CUDA_VISIBLE_DEVICES"] = '2'


class FaissModel():
    def __init__(self, embedding_file, faiss_file=None, sep='\t', rebuild=False, gpus=None) -> None:
        # embedding_file: [id] 文本 向量
        self.embedding_file = embedding_file
        self.sep = sep
        self.faiss_file = faiss_file if faiss_file else f"{embedding_file}.index"
        self.rebuild = rebuild
        
        self.ids_np = None
        self.embedding_np_l2 = None
        self.indexModel = None
        
        self.id2item = None  # id: item, 搜索后使用
        self.item2emb = None  # id: 向量, 内部搜索前使用，先根据item找emb
        self.emb_df = None
        self._read_csv()
        self._create_faiss_model(gpus=gpus)
    
    def _read_csv(self,):
        print("正在读取embedding文件: ", self.embedding_file)
        self.emb_df = pd.read_csv(self.embedding_file, sep=self.sep, header=None, index_col=None)  # [id], item, embedding
        
        if self.emb_df.shape[1] == 3:  # 有3列，使用自带的id
            id = self.emb_df[0]
            self.emb_df = self.emb_df.drop(0, axis=1)  # 为统一，删除id列
        elif self.emb_df.shape[1] == 2:  # 有2列，生成id
            id = range(len(self.emb_df))
        else:
            raise Exception(f"check emb_df: {self.emb_df.head(2)}")

        self.ids_np = np.array(id, dtype=np.int64)  # ids需要转成int64类型
        
        self.id2item = {id[index]: row[0] for index, row in self.emb_df.iterrows()}  # id: item
        
        embedding = list(map(lambda x: eval(x)[0] if len(eval(x)) == 1 else eval(x), self.emb_df[1]))
        embedding_np = np.array(embedding, dtype=np.float32)  # embedding需要转成float32类型，待搜索的embedding也是
        # print(embedding_np[0], embedding_np.shape)
        self.embedding_np_l2 = preprocessing.normalize(embedding_np, norm='l2')  # 按行，l2范数归一化
        
        self.item2emb = dict(zip(self.emb_df[0].values.tolist(), self.embedding_np_l2))

        print(f"{self.embedding_file} faiss模型共有 {self.ids_np.shape[0]} 条训练数据")
    
    def _read_faiss_model(self, ):
        r"""读取faiss索引文件
        """
        self.indexModel = faiss.read_index(self.faiss_file)
    
    def _save_faiss_model(self, ):
        faiss.write_index(self.indexModel, self.faiss_file)  # 保存模型
        print(f"faiss索引已保存在: {faiss_file}")
    
    def _create_faiss_model(self, gpus=None):
        if Path(self.faiss_file).exists() and not self.rebuild:
            print("faiss索引文件已存在")
            self._read_faiss_model()
            return
        else:
            print("开始构建faiss索引……")
        
        # 无论使用那个索引，必须使用内积，L2范数归一化后的内积代表余弦相似度
        dim, measure = self.embedding_np_l2.shape[1], faiss.METRIC_INNER_PRODUCT  # METRIC_L2, METRIC_INNER_PRODUCT

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
        self.indexModel = faiss.IndexHNSWFlat(dim, 64, measure)  # 第二个参数为构建图时每个点最多连接多少个节点，x越大，构图越复杂，查询越精确，当然构建index时间也就越慢，x取4~64中的任何一个整数
        self.indexModel.hnsw.efSearch = 64  # efSearch越大越准确，但是速度越慢（不支持GPU）
        self.indexModel = faiss.IndexIDMap(self.indexModel)
        
        # n_bits = 2 * dim
        # self.indexModel = faiss.IndexLSH(dim, n_bits)
        
        ngpus = faiss.get_num_gpus()
        print("number of GPUs:", ngpus)
        if isinstance(gpus, list):
            self.indexModel = faiss.index_cpu_to_gpus_list(self.indexModel, gpus=gpus)

        print(f"is_trained: {self.indexModel.is_trained}")
        if not self.indexModel.is_trained:  # bool类型，用于指示index是否已被训练
            self.indexModel.train(self.embedding_np_l2)  # 倒排索引需要训练k-means
        # indexModel.add(embedding) # add_with_ids已经包含add
        self.indexModel.add_with_ids(self.embedding_np_l2, self.ids_np)
        
        self._save_faiss_model()
    
    def search_item2item_one(self, query_item, query_nums=16, need_distance=False):
        r"""
        在训练的数据集中相互找相似item
        """
        query_emb = np.array([self.item2emb.get(query_item)])
        if query_emb is not None:
            Distance, Index = self.indexModel.search(query_emb, query_nums)
            if need_distance:
                return itemgetter(*Index[0])(self.id2item), Distance
            else:
                return itemgetter(*Index[0])(self.id2item)
        else:
            raise Exception(f"no item: {query_item}")
    
    def search_emb2item_one(self, query_emb, query_nums=16, need_distance=False):
        r"""
        根据另外的向量数据, 在训练集中找相似item
        """
        if isinstance(query_emb, str):
            query_emb = eval(query_emb)
        if isinstance(query_emb, list):
            query_emb = np.array(query_emb, dtype=np.float32)
        if len(query_emb.shape) != 2:
            query_emb.reshape(1, query_emb.shape[0])
        Distance, Index = self.indexModel.search(query_emb, query_nums)
        if need_distance:
            return itemgetter(*Index[0])(self.id2item), Distance
        else:
            return itemgetter(*Index[0])(self.id2item)
        
    def search_item2item_all(self, saved_similars_file=None):
        print("正在给整个embedding文件匹配相似item……")
        if not saved_similars_file:
            saved_similars_file = self.embedding_file[:-4] + "_similars.txt"
        
        tqdm.pandas()
        sens_pair_df = self.emb_df.copy()
        sens_pair_df[2] = self.emb_df[1].progress_apply(self.search_emb2item_one)  # 给原句匹配相似句
        # print(sens_pair_df.head())
        
        sens_pair_df.to_csv(saved_similars_file, sep=self.sep, encoding='utf_8_sig', header=False, index=False)
        
        print("相似item查找完毕, 在文件: ", saved_similars_file)


if __name__ == "__main__":
    # 根据embedding文件构建索引
    embedding_file = "vocabulary/char_emb_std.csv"  # id, item, embedding
    faiss_file = f"{embedding_file}.index"  # 要保存的faiss文件
    faiss_model = FaissModel(embedding_file, faiss_file)
    
    similar_item, distance = faiss_model.search_item2item_one("大", need_distance=True)
    print(similar_item)
    print(distance)
    
    # 大
    query_emb = [[-1626.141845703125, -258.3230285644531, -170.23858642578125, -453.48797607421875, -679.236083984375, -442.3926696777344, -623.2569580078125, -324.0648498535156, -691.6055908203125, -417.3226013183594, 9.010758399963379, -471.7956848144531, -1619.129150390625, -274.50048828125, -638.0111694335938, -447.400634765625, 201.26202392578125, -310.2436828613281, -708.9534301757812, -489.09808349609375, -618.6454467773438, -789.253662109375, -662.8973999023438, -773.4139404296875, -656.8958129882812, -418.7200622558594, -1596.4967041015625, -776.9157104492188, -17.862668991088867, -1574.9481201171875, -1646.557373046875, -450.7420654296875, -462.2167663574219, -662.0615234375, -884.4080810546875, -852.1771850585938, -1096.4381103515625, -445.37274169921875, -536.5611572265625, -726.61279296875, -1118.9298095703125, -815.8878173828125, -627.7644653320312, -614.3839111328125, -405.5152587890625, -1578.3507080078125, -653.5629272460938, -570.5039672851562, -746.6898803710938, -568.177978515625, -482.58477783203125, -586.5888061523438, -279.24444580078125, -399.5200500488281, -1177.607421875, -803.8395385742188, -452.0279846191406, -745.097412109375, -1260.422607421875, -1121.4434814453125, -991.6651000976562, -560.9479370117188, -434.57763671875, -859.7173461914062, 46.84480285644531, -550.7662963867188, -1593.549072265625, -540.8893432617188, -1647.0078125, -1627.7733154296875, -430.5883483886719, -1594.18017578125, -386.7716064453125, -936.0023803710938, -732.535400390625, -764.0670776367188, -726.2144775390625, -605.2793579101562, -793.26513671875, -871.6861572265625, -817.1549682617188, -665.3388671875, -726.0913696289062, -1594.1783447265625, -282.04412841796875, 99.20069122314453, -39.88848114013672, -20.332725524902344, -237.18829345703125, -696.5205688476562, -380.0174560546875, 981.5438232421875]]
    
    print("使用未归一化的向量搜索: ")
    similar_item, distance = faiss_model.search_emb2item_one(query_emb, need_distance=True)
    print(similar_item)
    print(distance)
    
    print("使用已归一化的向量搜索: ")
    query_emb = preprocessing.normalize(np.array(query_emb, dtype=np.float32), norm='l2')  # 按行，l2范数归一化
    similar_item, distance = faiss_model.search_emb2item_one(query_emb, need_distance=True)
    print(similar_item)
    print(distance)
    # 可以看到搜索时有没有归一化不影响找到的item元素，只影响相似距离的度量
    
    # faiss_model.search_item2item_all(saved_similars_file=embedding_file[:-4] + "_similars.txt")
