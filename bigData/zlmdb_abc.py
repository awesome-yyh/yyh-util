import os
import lmdb
from abc import ABC
from abc import abstractmethod


class ZLMDBABC(ABC):
    """lmdb 抽象基类
    子类需要实现io方法, 使其传入index后即可读/写
    """
    def __init__(self, lmdb_path, map_size=int(1e12)):
        
        os.makedirs(lmdb_path, exist_ok=True)
        self.db = lmdb.open(lmdb_path, map_size, lock=False, max_readers=126)
        
        self.__len__()
        
    def __len__(self, ):
        self.train_data_len = int(self._get("train_data_len")) if self._get("train_data_len") else 0
        
        return self.train_data_len
        
    def _put(self, k, v):
        """增加和修改都是这个
        """
        with self.db.begin(write=True) as txn:
            txn.put(key=str(k).encode(encoding='utf-8'), value=str(v).encode(encoding='utf-8'))
    
    def _get(self, k):
        """查询
        """
        with self.db.begin() as txn:
            try:
                v = txn.get(str(k).encode(encoding='utf-8'))  # 不存在时返回None
                if v:
                    v = str(v, encoding='utf-8')
                return v
            except:
                return None

    def _delete(self, key):
        """删除
        """
        with self.db.begin(write=True) as txn:
            txn.delete(key=str(key).encode(encoding='utf-8'))

    def display(self, n=10):
        """遍历前n项
        """
        with self.db.begin() as txn:
            for count, (key, value) in enumerate(txn.cursor()):
                print(str(key, encoding='utf-8'), str(value, encoding='utf-8'))
                if count >= n:
                    break
    
    @abstractmethod
    def put_io(sel, *args):
        """
        写入训练数据，并更新self.train_data_len
        """

    @abstractmethod
    def get_io(self, idx):
        """
        根据id获取元素，可以获取多个值后返回
        """
    
    def __del__(self,):
        self.db.close()
