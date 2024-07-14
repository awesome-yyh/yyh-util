"""将csv文件数据保存到lmdb"""
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from .zlmdb import ZLMDB


pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 100)
filename = "data/csv.train_0_clear.csv"

lmdb_dir = "data/train_clear"
db = ZLMDB(lmdb_path=lmdb_dir)
print("初始lmdb大小: ", len(db))
# db.display(5)


def process_row(row):
    # print(row)
    # print(row[0], row[1])
    db.put_io_choose(row[0], row[1])


print("正在处理文件: ", filename)
for chunk in tqdm.tqdm(pd.read_csv(filename, chunksize=1024, sep='\t', encoding='utf_8_sig', header=None, index_col=None, quoting=csv.QUOTE_NONE, on_bad_lines='skip'), total=46817 // 1024 + 1): 
    # print(chunk.head())
    chunk.apply(process_row, axis=1)
    # break


print("实存: ", len(db))
print(db.get_io_choose(0))
