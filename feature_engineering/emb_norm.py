import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm


# 对embedding文件进行标准化或归一化
test = True  # 小批量测试程序
processing_type = "norm"  # std or norm
csv_path = "vocabulary/char_emb.csv"
saved_csv_path = f"{csv_path[:-4]}_{processing_type}.csv"

df = pd.read_csv(csv_path, sep='\t', encoding='utf_8_sig', header=None, index_col=None)
print(df.head())

if test:
    df = df.head()


# ll = df[0].values.tolist()
# print(ll)

tqdm.pandas()

if processing_type == "std":
    print("正在进行标准化……")
    embedding_list = list(map(lambda x: eval(x)[0] if len(eval(x)) == 1 else eval(x), df[1]))
    embedding_list = np.array(embedding_list)

    mean = np.mean(embedding_list, axis=0)
    std = np.std(embedding_list, axis=0)
    print("mean: ", mean, "std: ", std)

    # 标准化(按列)
    df[1] = df[1].progress_apply(lambda x: [((eval(x)[0] - mean) / std).tolist()])
else:
    print("正在进行归一化……")
    # 归一化(按行)
    df[1] = df[1].progress_apply(lambda x: preprocessing.Normalizer(norm="l2").fit_transform(np.array(eval(x))).tolist())

df.to_csv(saved_csv_path, sep='\t', encoding='utf_8_sig', header=False, index=False)


# 测试保存的文件
df = pd.read_csv(saved_csv_path, sep='\t', encoding='utf_8_sig', header=None, index_col=None)

tqdm.pandas()
df[1] = df[1].progress_apply(lambda x: eval(x))
print(df.head())
