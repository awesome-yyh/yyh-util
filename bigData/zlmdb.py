import os
import lmdb


class ZLMDB():
    def __init__(self, lmdb_path, map_size=int(1e12)):
        
        os.makedirs(lmdb_path, exist_ok=True)
        self.db = lmdb.open(lmdb_path, map_size, lock=False, max_readers=126)  # lmdb.Environment的别名
        
        self.train_data_len = int(self.get("train_data_len")) if self.get("train_data_len") else 0
        print("train_data_len:", self.train_data_len)

    def _put(self, k, v):
        # 增加和修改都是这个
        with self.db.begin(write=True) as txn:
            txn.put(key=str(k).encode(), value=str(v).encode())

    def delete(self, key):
        # 删除
        with self.db.begin(write=True) as txn:
            txn.delete(key=str(key).encode())

    def get(self, k):
        # 查询
        with self.db.begin(write=True) as txn:
            v = txn.get(str(k).encode())
            if v:
                v = str(v, encoding='utf-8')
            return v

    def display(self, n=10):
        # 遍历前n项
        with self.db.begin() as txn:
            for count, (key, value) in enumerate(txn.cursor()):
                print(key, value)
                if count >= n:
                    break

    def put_io(self, instruction, input, output):
        self._put(f'instruction_{self.train_data_len}', instruction)
        self._put(f'input_{self.train_data_len}', input)
        self._put(f'output_{self.train_data_len}', output)
        
        self.train_data_len += 1
        self._put("train_data_len", self.train_data_len)

    def get_io(self, idx):
        instruction = self.get(f'instruction_{idx}')
        input = self.get(f'input_{idx}')
        output = self.get(f'output_{idx}')
        
        return instruction, input, output

    def __del__(self,):
        self.db.close()


if __name__ == '__main__':
    data_file = "data/train_lmdb/train_09"
    db = ZLMDB(lmdb_path=data_file)
    
    db.display(10)
    
    print(db.get("train_data_len"), type(db.get("train_data_len")))
    
    db.put_io('as', 'qw', 's')
    
    print(db.get_io(0))
