import csv
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn


# filename = "data/xxx.csv"
# df_train = pd.read_csv(filename, sep='\t', encoding='utf_8_sig', header=None, index_col=None, quoting=csv.QUOTE_NONE)

df_train = pd.DataFrame({0: ['A', 'B', 'A', 'C', 'B', 'A']})

print("------one-hot(pd)-------")
embarked_oht = pd.get_dummies(df_train[0])
Pclass_oht = pd.get_dummies(df_train[0].apply(lambda x: str(x)))  # 先转文本
print(pd.concat([df_train, Pclass_oht], axis=1))

print("------one-hot(sklearn)-------")
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df_train[[0]])
encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['color']))
print(encoded_data)

print("------二进制-------")

print("------标签-------")
label_encoder = LabelEncoder()
encoded_categories = label_encoder.fit_transform(df_train[[0]])
df_train['Encoded_Category'] = encoded_categories
print(df_train)


print("------embedding-------")
torch.random.manual_seed(42)
num_categories = 108
embedding_dim = 5
embedding = nn.Embedding(num_categories, embedding_dim)
category_feature = torch.LongTensor([[1, 63, 75, 22, 63], [1, 63, 75, 22, 63]])
embedded_feature = embedding(category_feature)
print(embedded_feature)
