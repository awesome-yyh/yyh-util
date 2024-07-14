from typing import List

try:
    from zlmdb_abc import ZLMDBABC
except:
    from .zlmdb_abc import ZLMDBABC


class ZLMDB(ZLMDBABC):
    def put_io(self, seq1: str, seq2: str):
        self._put(f"seq1_{self.train_data_len}", seq1)
        self._put(f"seq2_{self.train_data_len}", seq2)
        
        self.train_data_len += 1
        self._put("train_data_len", self.train_data_len)
    
    def get_io(self, idx):
        seq1 = self._get(f'seq1_{idx}')
        seq2 = self._get(f'seq2_{idx}')
        
        return seq1, seq2
    
    def put_emb(self, term, emb):
        self._put(f'term_emb_{term}', emb)
        
        self.train_data_len += 1
        self._put("train_data_len", self.train_data_len)
    
    def get_emb(self, term):
        emb = self._get(f'term_emb_{term}')
        if emb is not None:
            emb = eval(emb)[0]
        return emb

    def put_io_triplet(self, anchor: str, pos: str, neg: str):
        self._put(f"anchor_{self.train_data_len}", anchor)
        self._put(f"pos_{self.train_data_len}", pos)
        self._put(f"neg_{self.train_data_len}", neg)
        
        self.train_data_len += 1
        self._put("train_data_len", self.train_data_len)
    
    def get_io_triplet(self, idx):
        anchor = self._get(f'anchor_{idx}')
        pos = self._get(f'pos_{idx}')
        neg = self._get(f'neg_{idx}')
        
        return anchor, pos, neg


if __name__ == '__main__':
    data_file = "data/termonline_emb_lmdb_ptxxxx"
    db = ZLMDB(lmdb_path=data_file)
    print("len: ", len(db))
    
    # db.put_io('as', 'qw')
    
    # db.display(10)
    
    # print(db.get_io(0))
    # print(db.get_io(1))
    
    # print(db.get_emb("马尔可夫过程"))
    # print(db.get_emb("无障碍设计"))
    
    print(db.get_io_triplet(0))
    print(db.get_io_triplet(1))
